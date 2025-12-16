"""
Caso de uso para processamento de detecção de faces (Application Layer).
Orquestra o fluxo completo de detecção, tracking e envio ao FindFace.
"""

import logging
import threading
import gc
from queue import Queue, Empty
from typing import Dict, Optional, List, Tuple
from collections import defaultdict, deque
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
import numpy as np

from src.domain.entities import Camera, Track, Event
from src.domain.interfaces import IFaceDetector, ILandmarksDetector
from src.domain.services import (
    TrackValidationService,
    EventCreationService,
    MovementDetectionService,
    TrackLifecycleService,
    FaceQualityService,
    ImageSaveService
)
from src.domain.adapters import FindfaceAdapter
from src.domain.value_objects import IdVO, FullFrameVO, TimestampVO


class ProcessFaceDetectionUseCase:
    """
    Caso de uso para processar detecção de faces em streams de vídeo.
    Orquestra todas as operações: leitura de frames, detecção, tracking,
    validação e envio ao FindFace.
    """

    def __init__(
        self,
        camera: Camera,
        face_detector: IFaceDetector,
        findface_adapter: FindfaceAdapter,
        findface_queue: Queue,
        event_creation_service: EventCreationService,
        movement_detection_service: MovementDetectionService,
        track_validation_service: TrackValidationService,
        track_lifecycle_service: TrackLifecycleService,
        landmarks_detector: Optional[ILandmarksDetector] = None,
        gpu_id: int = 0,
        image_save_service: Optional[ImageSaveService] = None,
        face_quality_service: Optional[FaceQualityService] = None,
        tracker_config: str = "bytetrack.yaml",
        batch_size: int = 1,
        show_video: bool = False,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        landmark_conf_threshold: float = 0.1,
        landmark_iou_threshold: float = 0.75,
        max_frames_lost: int = 30,
        save_images: bool = False,
        project_dir: str = "imagens",
        results_dir: str = "rtsp_byte_track_results",
        min_movement_threshold: float = 10.0,
        min_movement_percentage: float = 50.0,
        min_confidence_threshold: float = 0.5,
        min_bbox_width: int = 50,
        max_frames_per_track: int = 100,
        inference_size: Optional[tuple] = None,
        detection_skip_frames: int = 0,
        jpeg_quality: int = 95,
        success_tracking_window: int = 100
    ):
        """
        Inicializa o caso de uso de processamento de detecção de faces.
        
        :param camera: Entidade Camera.
        :param face_detector: Implementação de IFaceDetector.
        :param findface_adapter: Adapter para envio ao FindFace.
        :param findface_queue: Fila global compartilhada do FindFace.
        :param event_creation_service: Serviço de criação de eventos (pré-configurado).
        :param movement_detection_service: Serviço de detecção de movimento.
        :param track_validation_service: Serviço de validação de tracks.
        :param track_lifecycle_service: Serviço de ciclo de vida de tracks.
        :param landmarks_detector: Implementação opcional de ILandmarksDetector (inferência síncrona em batch).
        :param gpu_id: ID da GPU utilizada.
        :param image_save_service: Serviço para salvamento assíncrono de imagens.
        :param face_quality_service: Serviço para cálculo de qualidade facial.
        :param tracker_config: Configuração do ByteTrack.
        :param show_video: Se deve exibir vídeo (debug).
        :param conf_threshold: Threshold de confiança mínima.
        :param iou_threshold: Threshold de IoU para NMS.
        :param landmark_conf_threshold: Threshold de confiança para detecção de landmarks.
        :param landmark_iou_threshold: Threshold de IoU para NMS de landmarks.
        :param max_frames_lost: Máximo de frames perdidos antes de finalizar track.
        :param save_images: Se deve salvar imagens.
        :param project_dir: Diretório do projeto.
        :param results_dir: Diretório de resultados.
        :param min_movement_threshold: Distância mínima para considerar movimento.
        :param min_movement_percentage: Percentual mínimo de frames com movimento.
        :param min_confidence_threshold: Confiança mínima para enviar.
        :param min_bbox_width: Largura mínima do bbox.
        :param max_frames_per_track: Máximo de eventos por track.
        :param inference_size: Tamanho de inferência.
        :param detection_skip_frames: Frames a pular entre detecções (não usado no modo YOLO stream).
        """
        # Dependências injetadas
        self.camera = camera
        self.face_detector = face_detector
        self.findface_adapter = findface_adapter
        self.findface_queue = findface_queue
        self.landmarks_detector = landmarks_detector
        self.gpu_id = gpu_id
        self.image_save_service = image_save_service
        
        # Domain Services (injetados já configurados)
        self.event_creation_service = event_creation_service
        self.movement_detection_service = movement_detection_service
        self.track_validation_service = track_validation_service
        self.track_lifecycle_service = track_lifecycle_service
        
        # Configurações
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.landmark_conf_threshold = landmark_conf_threshold
        self.landmark_iou_threshold = landmark_iou_threshold
        self.max_frames_lost = max_frames_lost
        self.save_images = save_images
        self.show_video = show_video
        self.project_dir = project_dir
        self.results_dir = results_dir
        self.jpeg_quality = jpeg_quality
        self.inference_size = inference_size
        self.detection_skip_frames = detection_skip_frames
        
        # Estado do processamento
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.active_tracks: Dict[int, Track] = {}
        self.track_frames_lost: Dict[int, int] = defaultdict(int)
        self.track_frame_count: Dict[int, int] = defaultdict(int)
        self.frame_count = 0
        
        # Rastreamento de taxa de sucesso FindFace (janela deslizante)
        self._findface_success_queue: deque = deque(maxlen=success_tracking_window)
        
        # OTIMIZAÇÃO 3: GC periódica - evita fragmentação de memória
        self._tracks_finalized_count = 0
        
        # Contador para warnings periódicos de fila cheia
        self._findface_queue_full_count = 0
        
        # CRÍTICO: Logger DEVE ser criado ANTES do worker thread
        self.logger = logging.getLogger(
            f"{self.__class__.__name__}-{camera.camera_name.value()}"
        )
        
        # OTIMIZAÇÃO: Buffer preallocado para bboxes (evita alocações repetidas)
        self.bbox_buffer = np.empty(4, dtype=np.float32)
        
        if self.landmarks_detector is not None:
            self.logger.info(
                f"Detector de landmarks configurado para inferência SÍNCRONA em batch"
            )
        
        # Log de configuração de salvamento de imagens
        if self.save_images and self.image_save_service is not None:
            self.logger.info(
                f"Salvamento de imagens HABILITADO - "
                f"Diretório: {self.project_dir}/{self.results_dir}/{self.camera.camera_name.value()} "
                f"(JPEG quality: {self.jpeg_quality}%)"
            )
        else:
            self.logger.info("Salvamento de imagens DESABILITADO")

    def start(self) -> None:
        """
        Inicia o processamento do stream de vídeo.
        """
        self.running = True
        self.logger.info(
            f"Iniciando processamento para câmera {self.camera.camera_name.value()} "
            f"(ID: {self.camera.camera_id.value()})"
        )
        
        try:
            self._process_stream()
        except Exception as e:
            self.logger.error(f"Erro no processamento do stream: {e}", exc_info=True)
        finally:
            self.stop()

    def stop(self) -> None:
        """
        Para o processamento do stream.
        """
        self.running = False
        
        # Finaliza serviço de salvamento de imagens
        if self.image_save_service is not None:
            try:
                self.image_save_service.stop()
                self.logger.info("ImageSaveService finalizado com sucesso")
            except Exception as e:
                self.logger.error(f"Erro ao finalizar ImageSaveService: {e}")
        
        self.logger.info(
            f"ProcessFaceDetectionUseCase finalizado para câmera "
            f"{self.camera.camera_name.value()}"
        )

    def get_success_rate(self) -> float:
        """
        Calcula a taxa de sucesso de envio ao FindFace.
        Considera apenas falhas de fila cheia (Queue.Full).
        
        :return: Taxa de sucesso (0.0 a 1.0), ou 1.0 se não houver dados.
        """
        if len(self._findface_success_queue) == 0:
            return 1.0
        
        success_count = sum(1 for success in self._findface_success_queue if success)
        return success_count / len(self._findface_success_queue)

    def _process_stream(self) -> None:
        """
        Processa o stream de vídeo diretamente pelo YOLO.
        O YOLO gerencia a leitura de frames internamente.
        """
        # YOLO gerencia o stream internamente
        for result in self.face_detector.detect_and_track(
            source=self.camera.source.value(),
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            tracker=self.tracker_config,
            inference_size=self.inference_size,
            show=self.show_video
        ):
            if not self.running:
                break
            
            self.frame_count += 1
            
            # OTIMIZAÇÃO: Processa detecções a cada N frames (economiza processamento)
            # Aplica APÓS inferência do YOLO e ANTES de criar eventos
            # Exemplo: se detection_skip_frames=2, processa frames 1, 3, 5, 7... (a cada 2 frames)
            if self.detection_skip_frames > 1:
                if self.frame_count % self.detection_skip_frames != 1:
                    # Pula processamento deste frame, mas mantém tracking ativo
                    continue
            
            # Obtém frame do resultado YOLO
            frame = result.orig_img
            timestamp = datetime.now()
            
            # Cria FullFrameVO com timestamp
            frame_vo = FullFrameVO(
                ndarray=frame,
                copy=False,  # Performance otimizada
                timestamp=TimestampVO(timestamp)
            )
            
            # Processa resultado da detecção
            try:
                self._process_detection_result(result, frame_vo)
            except Exception as e:
                self.logger.error(f"Erro ao processar detecção: {e}", exc_info=True)
                continue

    def _process_detection_result(self, result, frame_vo: FullFrameVO) -> None:
        """
        Processa resultado da detecção YOLO.
        
        :param result: Resultado da detecção YOLO.
        :param frame_vo: FullFrameVO com o frame.
        """
        # Extrai dados do resultado YOLO
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            self._update_lost_tracks()
            return
        
        # Verifica se landmarks já vieram da primeira inferência (YOLO com keypoints)
        has_keypoints_from_yolo = hasattr(result, 'keypoints') and result.keypoints is not None
        
        # IDs dos tracks detectados neste frame
        detected_track_ids = set()
        
        # OTIMIZAÇÃO: Batch inference de landmarks
        # Coleta todos os crops de face primeiro
        face_crops: List[np.ndarray] = []
        track_ids: List[int] = []
        bboxes: List[np.ndarray] = []
        confidences: List[float] = []
        yolo_landmarks: List = []  # Landmarks da primeira inferência (se existirem)
        
        h, w = frame_vo.ndarray_readonly.shape[:2]
        
        for i, box in enumerate(boxes):
            track_id = int(box.id[0]) if box.id is not None else None
            if track_id is None:
                continue
            
            # Adiciona track_id ao set de detectados ANTES dos filtros
            # Isso mantém o track ativo mesmo quando frames individuais não passam nos filtros
            detected_track_ids.add(track_id)
            
            # FILTRO 1: Verifica confiança mínima (filtragem no frame)
            # Se não passar, pula criação de evento mas mantém track ativo
            confidence = float(box.conf[0])
            if confidence < self.track_validation_service.min_confidence_threshold:
                continue
            
            # OTIMIZAÇÃO: Usa buffer preallocado para bbox
            np.copyto(self.bbox_buffer, box.xyxy[0].cpu().numpy())
            x1, y1, x2, y2 = self.bbox_buffer.astype(int)
            
            # Valida limites
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # FILTRO 2: Verifica largura mínima do bbox (filtragem no frame)
            bbox_width = x2 - x1
            if bbox_width < self.track_validation_service.min_bbox_width:
                continue
            
            # Crop da face
            face_crop = frame_vo.ndarray_readonly[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                face_crops.append(face_crop)
                track_ids.append(track_id)
                bboxes.append(self.bbox_buffer.copy())
                confidences.append(confidence)
                
                # Extrai landmarks da primeira inferência se disponível
                if has_keypoints_from_yolo:
                    keypoints_xy = result.keypoints.xy[i].cpu().numpy()
                    yolo_landmarks.append(keypoints_xy)
                else:
                    yolo_landmarks.append(None)
        
        # Decide se precisa fazer segunda inferência para landmarks
        landmarks_results = []
        
        if has_keypoints_from_yolo:
            # Usa landmarks da primeira inferência (YOLO)
            for i, landmarks_array in enumerate(yolo_landmarks):
                if landmarks_array is not None and len(landmarks_array) >= 5:
                    landmarks_results.append((landmarks_array, 1.0))
                else:
                    landmarks_results.append(None)
        
        elif self.landmarks_detector is not None and len(face_crops) > 0:
            # Landmarks NÃO vieram da primeira inferência - faz segunda inferência em BATCH
            try:
                batch_landmarks = self.landmarks_detector.predict_batch(
                    face_crops=face_crops,
                    conf=self.landmark_conf_threshold,
                    verbose=False
                )
                
                # Converte para formato esperado: List[Tuple[np.ndarray, float] | None]
                for landmarks_data in batch_landmarks:
                    if landmarks_data is not None:
                        landmarks_array, conf = landmarks_data
                        if landmarks_array is not None and len(landmarks_array) >= 5:
                            landmarks_results.append((landmarks_array, conf))
                        else:
                            landmarks_results.append(None)
                    else:
                        landmarks_results.append(None)
            except Exception as e:
                self.logger.error(f"Erro na segunda inferência de landmarks: {e}", exc_info=True)
                landmarks_results = [None] * len(face_crops)
        else:
            # Sem landmarks disponíveis
            landmarks_results = [None] * len(face_crops)
        
        # Cria eventos para todas as detecções
        for i, (track_id, bbox, confidence, landmarks_result) in enumerate(
            zip(track_ids, bboxes, confidences, landmarks_results)
        ):
            # Extrai crop do bbox para thumbnail
            face_crop = face_crops[i] if i < len(face_crops) else None
            
            # Cria evento a partir da detecção
            event = self.event_creation_service.create_event_from_detection(
                camera=self.camera,
                detection_box=bbox,
                confidence=confidence,
                track_id=track_id,
                keypoints=landmarks_result,
                frame_entity=frame_vo,
                index=i
            )
            
            # Armazena crop no evento para uso posterior (salvamento de thumbnail)
            # Criamos um atributo dinâmico para evitar modificar a entidade Event
            event._face_crop = face_crop
            
            # Cria ou recupera track
            if track_id not in self.active_tracks:
                # Cria novo track com o primeiro evento
                self.active_tracks[track_id] = Track(
                    id=IdVO(track_id),
                    first_event=event,
                    min_movement_percentage=self.track_validation_service.min_movement_percentage / 100.0
                )
                self.track_frames_lost[track_id] = 0
                self.track_frame_count[track_id] = 0
            
            track = self.active_tracks[track_id]
            self.track_frames_lost[track_id] = 0
            self.track_frame_count[track_id] += 1
            
            # Adiciona evento ao track (se não for o primeiro)
            if track.event_count > 0:
                added = self.track_lifecycle_service.add_event_to_track(track, event)
                
                if added:
                    self.logger.debug(
                        f"Track {track_id}: Evento adicionado "
                        f"(total: {track.event_count}, conf: {event.confidence.value():.2f})"
                    )
        
        # Atualiza tracks perdidos
        self._update_lost_tracks(detected_track_ids)

    def _update_lost_tracks(self, detected_track_ids: Optional[set] = None) -> None:
        """
        Atualiza contador de frames perdidos e finaliza tracks inativos.
        Também finaliza tracks que atingiram o limite máximo de eventos.
        
        :param detected_track_ids: Set de IDs detectados neste frame (ou None).
        """
        if detected_track_ids is None:
            detected_track_ids = set()
        
        tracks_to_remove = []
        
        for track_id in list(self.active_tracks.keys()):
            track = self.active_tracks[track_id]
            
            if track_id not in detected_track_ids:
                # Track não foi detectado - incrementa contador de frames perdidos
                self.track_frames_lost[track_id] += 1
            
            # Verifica se deve finalizar track (por frames perdidos OU por limite de eventos)
            if self.track_lifecycle_service.should_finalize_track(
                track=track,
                frames_lost=self.track_frames_lost[track_id],
                max_frames_lost=self.max_frames_lost
            ):
                tracks_to_remove.append(track_id)
        
        # Finaliza tracks
        for track_id in tracks_to_remove:
            self._finalize_track(track_id)

    def _finalize_track(self, track_id: int) -> None:
        """
        Finaliza um track e envia ao FindFace se válido.
        
        :param track_id: ID do track a finalizar.
        """
        track = self.active_tracks.get(track_id)
        if track is None:
            return
        
        # Coleta informações do track para log
        summary = self.track_lifecycle_service.get_track_summary(track)
        has_movement = track.has_movement
        total_events = summary['event_count']
        best_quality = summary['best_quality']
        best_confidence = summary['best_confidence']
        movement_stats = track.get_movement_statistics()
        
        # Valida track
        is_valid, invalid_reason = self.track_validation_service.is_valid(track)
        
        # Obtém melhor evento para salvamento (válido ou não)
        best_event = self.track_lifecycle_service.get_best_event(track)
        
        if is_valid:
            if best_event is not None:
                # SALVAMENTO ASSÍNCRONO: Salva fullframe com bbox desenhado (bbox verde para tracks válidos)
                if self.save_images and self.image_save_service is not None:
                    self._save_event_image_async(best_event, track_id, is_valid=True)
                
                # Enfileira para envio ao FindFace
                try:
                    self.findface_queue.put_nowait((
                        self.camera.camera_id.value(),
                        self.camera.camera_name.value(),
                        track_id,
                        best_event,
                        track.event_count
                    ))
                    # Sucesso no envio
                    self._findface_success_queue.append(True)
                except Exception as e:
                    # Fila cheia - contabiliza como falha
                    self._findface_success_queue.append(False)
                    self._findface_queue_full_count += 1
                    if self._findface_queue_full_count % 100 == 0:
                        self.logger.warning(
                            f"⚠ Fila de FindFace CHEIA - {self._findface_queue_full_count} ocorrências "
                            f"(tamanho: {self.findface_queue.qsize()}/{self.findface_queue.maxsize})"
                        )
                    self.logger.debug(f"Erro ao enfileirar evento FindFace: {e}")
        else:            
            # Log de track descartado (sempre registra)
            self.logger.warning(
                f"✗ Track {track_id} descartado | "
                f"Razão: {invalid_reason}, "
                f"Eventos: {total_events}, "
                f"Movimento: {'Sim' if has_movement else 'Não'}, "
                f"Qualidade: {best_quality:.4f}, "
                f"Confiança: {best_confidence:.2f}"
            )
            
            # SALVAMENTO ASSÍNCRONO: Salva fullframe com bbox vermelho para tracks descartados
            if best_event is not None and self.save_images and self.image_save_service is not None:
                self._save_event_image_async(best_event, track_id, is_valid=False)
        
        # Log DEBUG com estatísticas completas do track finalizado
        self.logger.debug(
            f"Track {track_id} finalizado | "
            f"Válido: {'Sim' if is_valid else 'Não'}, "
            f"Eventos: {summary['event_count']}, "
            f"Movimento: {summary['has_movement']}, "
            f"Distância média: {summary['average_distance']:.2f}px, "
            f"Distância máxima: {summary['max_distance']:.2f}px, "
            f"% frames com movimento: {movement_stats['movement_percentage']:.1f}%, "
            f"Melhor confiança: {summary['best_confidence']:.2f}, "
            f"Melhor qualidade: {summary['best_quality']:.4f}"
        )

        # Remove track
        del self.active_tracks[track_id]
        if track_id in self.track_frames_lost:
            del self.track_frames_lost[track_id]
        if track_id in self.track_frame_count:
            del self.track_frame_count[track_id]
        
        # OTIMIZAÇÃO #3: GC periódica (a cada 500 tracks finalizados)
        # Evita fragmentação de memória e pausas longas de GC
        self._tracks_finalized_count += 1
        if self._tracks_finalized_count % 500 == 0:
            gc.collect()
    
    def _save_event_image_async(self, event: Event, track_id: int, is_valid: bool = True) -> None:
        """
        Salva fullframe com bbox do evento desenhado e thumbnail com landmarks (assíncrono).
        
        :param event: Evento com o melhor frame.
        :param track_id: ID do track para nomenclatura.
        :param is_valid: Se True, desenha bbox verde; se False, desenha bbox vermelho.
        """
        try:
            import cv2
            import numpy as np
            from pathlib import Path
            
            self.logger.debug(f"Iniciando preparação de salvamento para track {track_id}")
            
            # Obtém fullframe e bbox
            full_frame = event.frame.full_frame.value().copy()  # Copia para não modificar original
            bbox = event.bbox.value()  # (x1, y1, x2, y2)
            
            # Define cor do bbox: verde para válidos, vermelho para descartados
            bbox_color = (0, 255, 0) if is_valid else (0, 0, 255)  # BGR: verde ou vermelho
            
            # Desenha bbox com espessura 2
            x1, y1, x2, y2 = bbox
            cv2.rectangle(full_frame, (x1, y1), (x2, y2), bbox_color, 2)
            
            # Adiciona label com track_id, confiança e qualidade
            # Formato: 'Track: XX, C: YY, Q: ZZ'
            confidence = event.confidence.value()
            quality = event.face_quality_score.value()
            label = f"Track: {track_id}, C: {confidence:.2f}, Q: {quality:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y1 - 10, label_size[1] + 10)
            
            # Background para texto
            cv2.rectangle(
                full_frame,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0], label_y + 5),
                bbox_color,
                -1
            )
            
            # Texto
            cv2.putText(
                full_frame,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
            
            # Cria path de salvamento
            # Formato: project_dir/results_dir/camera_name/track_XXXXX_timestamp.jpg
            timestamp = event.frame.timestamp.value()
            # Formata timestamp como string (YYYYMMDD_HHMMSS)
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Remove últimos 3 dígitos de microsegundos
            camera_name = self.camera.camera_name.value()
            
            output_dir = Path(self.project_dir) / self.results_dir / camera_name
            filename = f"track_{track_id:05d}_{timestamp_str}.jpg"
            filepath = output_dir / filename
            
            # Enfileira para salvamento assíncrono do fullframe
            success = self.image_save_service.save_async(
                image=full_frame,
                filepath=filepath,
                jpeg_quality=self.jpeg_quality
            )
            
            if not success:
                self.logger.warning(f"✗ Falha ao enfileirar imagem fullframe do track {track_id} (fila cheia)")
            
            # Salva thumbnail com landmarks desenhados (se disponível)
            face_crop = getattr(event, '_face_crop', None)
            landmarks = event.landmarks.value()
            
            if face_crop is not None and landmarks is not None and len(landmarks) > 0:
                # Copia crop para não modificar original
                crop_with_landmarks = face_crop.copy()
                
                # Define cores diferentes para cada landmark (5 pontos)
                # Cores em BGR: azul, verde, vermelho, ciano, magenta
                landmark_colors = [
                    (255, 0, 0),    # Azul
                    (0, 255, 0),    # Verde
                    (0, 0, 255),    # Vermelho
                    (255, 255, 0),  # Ciano
                    (255, 0, 255)   # Magenta
                ]
                
                # Desenha cada ponto com cor diferente
                for idx, (x, y) in enumerate(landmarks):
                    color = landmark_colors[idx % len(landmark_colors)]
                    # Desenha círculo sólido de raio 3 pixels
                    cv2.circle(crop_with_landmarks, (int(x), int(y)), 3, color, -1)
                    # Desenha borda preta ao redor para melhor visibilidade
                    cv2.circle(crop_with_landmarks, (int(x), int(y)), 3, (0, 0, 0), 1)
                
                # Nome do arquivo com sufixo '_crop'
                filename_crop = f"track_{track_id:05d}_{timestamp_str}_crop.jpg"
                filepath_crop = output_dir / filename_crop
                
                # Enfileira para salvamento assíncrono do crop
                success_crop = self.image_save_service.save_async(
                    image=crop_with_landmarks,
                    filepath=filepath_crop,
                    jpeg_quality=self.jpeg_quality
                )
                
                if not success_crop:
                    self.logger.warning(f"✗ Falha ao enfileirar imagem crop do track {track_id} (fila cheia)")
            
        except Exception as e:
            self.logger.error(f"Erro ao preparar salvamento de imagem do track {track_id}: {e}")
            self.logger.debug(f"GC executado após {self._tracks_finalized_count} tracks finalizados")
