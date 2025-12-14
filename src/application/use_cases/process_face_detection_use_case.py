"""
Caso de uso para processamento de detecção de faces (Application Layer).
Orquestra o fluxo completo de detecção, tracking e envio ao FindFace.
"""

import logging
import threading
import gc
from queue import Queue, Empty
from typing import Dict, Optional, List, Tuple
from collections import defaultdict
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
        max_frames_lost: int = 30,
        verbose_log: bool = False,
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
        jpeg_quality: int = 95
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
        :param max_frames_lost: Máximo de frames perdidos antes de finalizar track.
        :param verbose_log: Se deve fazer log detalhado.
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
        self.max_frames_lost = max_frames_lost
        self.verbose_log = verbose_log
        self.save_images = save_images
        self.show_video = show_video
        self.project_dir = project_dir
        self.results_dir = results_dir
        self.jpeg_quality = jpeg_quality
        self.inference_size = inference_size
        self.detection_skip_frames = detection_skip_frames
        
        # Estado do processamento
        self.running = False
        self.active_tracks: Dict[int, Track] = {}
        self.track_frames_lost: Dict[int, int] = defaultdict(int)
        self.track_frame_count: Dict[int, int] = defaultdict(int)
        self.frame_count = 0
        
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
            inference_size=self.inference_size,
            show=self.show_video
        ):
            if not self.running:
                break
            
            self.frame_count += 1
            
            # OTIMIZAÇÃO: Skip frames para economizar processamento
            # Aplica APÓS inferência do YOLO e ANTES de criar eventos
            if self.detection_skip_frames > 0:
                if self.frame_count % (self.detection_skip_frames + 1) != 0:
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
        
        if has_keypoints_from_yolo:
            self.logger.debug(f"✓ YOLO retornou keypoints na primeira inferência")
        else:
            self.logger.debug(f"✗ YOLO NÃO retornou keypoints - será necessária segunda inferência")
        
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
            
            detected_track_ids.add(track_id)
            
            # OTIMIZAÇÃO: Usa buffer preallocado para bbox
            np.copyto(self.bbox_buffer, box.xyxy[0].cpu().numpy())
            x1, y1, x2, y2 = self.bbox_buffer.astype(int)
            
            # Valida limites
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Crop da face
            face_crop = frame_vo.ndarray_readonly[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                face_crops.append(face_crop)
                track_ids.append(track_id)
                bboxes.append(self.bbox_buffer.copy())
                confidences.append(float(box.conf[0]))
                
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
            self.logger.info(f"✓ Usando landmarks da primeira inferência YOLO ({len(yolo_landmarks)} detecções)")
            for i, landmarks_array in enumerate(yolo_landmarks):
                if landmarks_array is not None and len(landmarks_array) >= 5:
                    # Confiança 1.0 pois veio da detecção principal
                    landmarks_results.append((landmarks_array, 1.0))
                    self.logger.debug(f"  - Face {i}: {len(landmarks_array)} landmarks")
                else:
                    landmarks_results.append(None)
                    self.logger.warning(f"  - Face {i}: landmarks insuficientes ou None")
        
        elif self.landmarks_detector is not None and len(face_crops) > 0:
            # Landmarks NÃO vieram da primeira inferência - faz segunda inferência em BATCH
            self.logger.info(f"⚙ Executando segunda inferência em batch para landmarks ({len(face_crops)} crops)")
            try:
                batch_landmarks = self.landmarks_detector.predict_batch(
                    face_crops=face_crops,
                    conf=0.5,
                    verbose=False
                )
                
                self.logger.debug(f"Segunda inferência retornou {len(batch_landmarks)} resultados")
                
                # Converte para formato esperado: List[Tuple[np.ndarray, float] | None]
                for i, landmarks_data in enumerate(batch_landmarks):
                    if landmarks_data is not None:
                        landmarks_array, conf = landmarks_data
                        if landmarks_array is not None and len(landmarks_array) >= 5:
                            landmarks_results.append((landmarks_array, conf))
                            self.logger.debug(f"  - Face {i}: {len(landmarks_array)} landmarks (conf={conf:.2f})")
                        else:
                            landmarks_results.append(None)
                            self.logger.warning(f"  - Face {i}: landmarks insuficientes")
                    else:
                        landmarks_results.append(None)
                        self.logger.warning(f"  - Face {i}: predict_batch retornou None")
            except Exception as e:
                self.logger.error(f"Erro na segunda inferência de landmarks: {e}", exc_info=True)
                landmarks_results = [None] * len(face_crops)
        else:
            # Sem landmarks disponíveis
            if self.landmarks_detector is None:
                self.logger.warning("⚠ Detector de landmarks não configurado")
            landmarks_results = [None] * len(face_crops)
            self.logger.debug(f"Sem landmarks para {len(face_crops)} faces")
        
        # Cria eventos para todas as detecções
        for i, (track_id, bbox, confidence, landmarks_result) in enumerate(
            zip(track_ids, bboxes, confidences, landmarks_results)
        ):
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
                
                if self.verbose_log and added:
                    self.logger.debug(
                        f"Track {track_id}: Evento adicionado "
                        f"(total: {track.event_count}, conf: {event.confidence.value():.2f})"
                    )
        
        # Atualiza tracks perdidos
        self._update_lost_tracks(detected_track_ids)

    def _update_lost_tracks(self, detected_track_ids: Optional[set] = None) -> None:
        """
        Atualiza contador de frames perdidos e finaliza tracks inativos.
        
        :param detected_track_ids: Set de IDs detectados neste frame (ou None).
        """
        if detected_track_ids is None:
            detected_track_ids = set()
        
        tracks_to_remove = []
        
        for track_id in list(self.active_tracks.keys()):
            if track_id not in detected_track_ids:
                self.track_frames_lost[track_id] += 1
                
                # Verifica se deve finalizar track
                if self.track_lifecycle_service.should_finalize_track(
                    track=self.active_tracks[track_id],
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
        
        # Valida track
        is_valid, invalid_reason = self.track_validation_service.is_valid(track)
        
        if is_valid:
            # Obtém melhor evento
            best_event = self.track_lifecycle_service.get_best_event(track)
            if best_event is not None:
                # SALVAMENTO ASSÍNCRONO: Salva fullframe com bbox desenhado
                if self.save_images and self.image_save_service is not None:
                    self._save_event_image_async(best_event, track_id)
                
                # Enfileira para envio ao FindFace
                try:
                    self.findface_queue.put_nowait((
                        self.camera.camera_id.value(),
                        self.camera.camera_name.value(),
                        track_id,
                        best_event,
                        track.event_count
                    ))
                    
                    # Log de track enviado (apenas em modo verbose)
                    if self.verbose_log:
                        self.logger.info(
                            f"✓ Track {track_id} enviado para a fila do Findface | "
                            f"Eventos: {total_events}, "
                            f"Movimento: {'Sim' if has_movement else 'Não'}, "
                            f"Qualidade: {best_quality:.4f}, "
                            f"Confiança: {best_confidence:.2f}"
                        )
                except Exception as e:
                    # Fila cheia - log periódico a cada 100 ocorrências
                    self._findface_queue_full_count += 1
                    if self._findface_queue_full_count % 100 == 0:
                        self.logger.warning(
                            f"⚠ Fila de FindFace CHEIA - {self._findface_queue_full_count} ocorrências "
                            f"(tamanho: {self.findface_queue.qsize()}/{self.findface_queue.maxsize})"
                        )
                    if self.verbose_log:
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
    
    def _save_event_image_async(self, event: Event, track_id: int) -> None:
        """
        Salva fullframe com bbox do evento desenhado (assíncrono).
        
        :param event: Evento com o melhor frame.
        :param track_id: ID do track para nomenclatura.
        """
        try:
            import cv2
            from pathlib import Path
            
            self.logger.debug(f"Iniciando preparação de salvamento para track {track_id}")
            
            # Obtém fullframe e bbox
            full_frame = event.frame.full_frame.value().copy()  # Copia para não modificar original
            bbox = event.bbox.value()  # (x1, y1, x2, y2)
            
            # Desenha bbox verde (RGB: 0, 255, 0) com espessura 2
            x1, y1, x2, y2 = bbox
            cv2.rectangle(full_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Adiciona label com track_id e confiança
            label = f"Track {track_id} - Conf: {event.confidence.value():.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y1 - 10, label_size[1] + 10)
            
            # Background para texto
            cv2.rectangle(
                full_frame,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0], label_y + 5),
                (0, 255, 0),
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
            
            # Enfileira para salvamento assíncrono
            success = self.image_save_service.save_async(
                image=full_frame,
                filepath=filepath,
                jpeg_quality=self.jpeg_quality
            )
            
            if success:
                self.logger.info(f"✓ Imagem enfileirada para salvamento: {filename}")
            else:
                self.logger.warning(f"✗ Falha ao enfileirar imagem do track {track_id} (fila cheia)")
            
        except Exception as e:
            self.logger.error(f"Erro ao preparar salvamento de imagem do track {track_id}: {e}")
            if self.verbose_log:
                self.logger.debug(f"GC executado após {self._tracks_finalized_count} tracks finalizados")
