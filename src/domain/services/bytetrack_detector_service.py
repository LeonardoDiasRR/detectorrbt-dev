"""
Serviço de detecção e rastreamento de faces usando ByteTrack no domínio.
"""

# built-in
from typing import Optional, Dict, List
from collections import defaultdict
from datetime import datetime
import logging
import time
from queue import Queue
from threading import Thread

# 3rd party
import numpy as np
import cv2

# local
from src.domain.adapters.findface_adapter import FindfaceAdapter
from src.domain.entities import Camera, Frame, Event, Track
from src.domain.value_objects import IdVO, BboxVO, ConfidenceVO, LandmarksVO, TimestampVO, FullFrameVO
from src.domain.services.model_interface import IDetectionModel
from src.domain.services.landmarks_model_interface import ILandmarksModel
from src.domain.services.image_save_service import ImageSaveService


class ByteTrackDetectorService:
    """
    Serviço de domínio responsável por detectar e rastrear faces em streams de vídeo.
    Utiliza entidades de domínio (Camera, Frame, Event, Track) seguindo princípios DDD.
    """

    def __init__(
        self,
        camera: Camera,
        detection_model: IDetectionModel,  # ALTERADO de yolo_model
        landmarks_model: Optional[ILandmarksModel] = None,  # NOVO: Modelo para landmarks faciais
        findface_adapter: Optional[FindfaceAdapter] = None,
        findface_queue: Optional[Queue] = None,  # NOVO: Fila FindFace global compartilhada
        image_save_service: Optional[ImageSaveService] = None,  # NOVO: Serviço assíncrono de salvamento
        tracker: str = "bytetrack.yaml",
        batch: int = 4,
        show: bool = True,
        stream: bool = True,
        conf: float = 0.1,
        iou: float = 0.2,
        max_frames_lost: int = 30,
        verbose_log: bool = False,
        save_images: bool = True,
        project_dir: str = "./imagens/",
        results_dir: str = "rtsp_byte_track_results",
        min_movement_threshold: float = 50.0,
        min_movement_percentage: float = 0.1,
        min_confidence_threshold: float = 0.45,
        min_bbox_width: int = 60,
        max_frames_per_track: int = 900,  # RENOMEADO
        inference_size: int = 640,  # NOVO: Tamanho da imagem para inferência
        detection_skip_frames: int = 1  # NOVO: Detectar a cada N frames (tracking continua em todos)
    ):
        """
        Inicializa o serviço de detecção de faces.

        :param camera: Entidade Camera com informações da câmera.
        :param detection_model: Modelo YOLO para detecção de faces.
        :param landmarks_model: Modelo para detecção de landmarks faciais (opcional).
        :param findface_adapter: Adapter para comunicação com FindFace (opcional).
        :param findface_queue: Fila FindFace global compartilhada entre câmeras (opcional).
        :param image_save_service: Serviço assíncrono de salvamento de imagens (opcional).
        :param tracker: Arquivo de configuração do tracker ByteTrack.
        :param batch: Tamanho do batch para processamento.
        :param show: Se deve exibir o vídeo processado.
        :param stream: Se deve processar em modo stream.
        :param conf: Threshold de confiança para detecções.
        :param iou: Threshold de IOU para o tracker.
        :param max_frames_lost: Máximo de frames perdidos antes de finalizar um track.
        :param verbose_log: Se deve exibir logs detalhados.
        :param project_dir: Diretório base para salvamento de imagens.
        :param results_dir: Nome do subdiretório para resultados.
        :param min_movement_threshold: Limite mínimo de movimento em pixels.
        :param min_movement_percentage: Percentual mínimo de frames com movimento (0.0 a 1.0).
        :param min_confidence_threshold: Confiança mínima para considerar track válido.
        :param min_bbox_width: Largura mínima do bbox do melhor evento (em pixels).
        :param max_frames_per_track: Máximo de frames permitidos por track.
        :param inference_size: Tamanho da imagem para inferência (ex: 640, 1280).
        :param detection_skip_frames: Realiza detecção a cada N frames (tracking continua em todos os frames).
        :param findface_queue_size: Tamanho da fila assíncrona para envios FindFace (0 = desabilita fila).
        :raises TypeError: Se camera não for do tipo Camera.
        """
        if not isinstance(camera, Camera):
            raise TypeError(f"camera deve ser Camera, recebido: {type(camera).__name__}")
        
        if findface_adapter is not None and not isinstance(findface_adapter, FindfaceAdapter):
            raise TypeError(f"findface_adapter deve ser FindfaceAdapter, recebido: {type(findface_adapter).__name__}")
        
        # Suprime warnings do OpenCV
        cv2.setLogLevel(0)
        
        self.camera = camera
        self.model = detection_model  # ALTERADO
        self.landmarks_model = landmarks_model  # NOVO: Modelo de landmarks
        self.findface_adapter = findface_adapter
        self.image_save_service = image_save_service  # NOVO: Serviço de salvamento assíncrono
        self.tracker = tracker
        self.batch = batch
        self.show = show
        self.stream = stream
        self.conf = conf
        self.iou = iou
        self.max_frames_lost = max_frames_lost
        self.verbose_log = verbose_log
        self.save_images = save_images
        self.project_dir = project_dir
        self.results_dir = results_dir
        self.min_movement_threshold = min_movement_threshold
        self.min_movement_percentage = min_movement_percentage
        self.min_confidence_threshold = min_confidence_threshold
        self.min_bbox_width = min_bbox_width
        self.max_frames_per_track = max_frames_per_track  # RENOMEADO
        self.inference_size = inference_size  # NOVO
        self.detection_skip_frames = max(1, detection_skip_frames)  # NOVO: mínimo 1
        self.running = False
        
        self.logger = logging.getLogger(
            f"ByteTrackDetectorService_{camera.camera_id.value()}_{camera.camera_name.value()}"
        )
        
        # Estruturas para rastreamento de tracks usando entidades do domínio
        self.active_tracks: Dict[int, Track] = {}
        self.track_frames_lost: Dict[int, int] = defaultdict(int)
        self.track_frame_count: Dict[int, int] = defaultdict(int)  # NOVO: contador de frames por track
        
        # Contador global de IDs para frames e eventos
        self._frame_id_counter = 0
        self._event_id_counter = 0
        
        # OTIMIZAÇÃO 7: Contador para coleta de lixo periódica
        self._tracks_finalized_count = 0
        
        # OTIMIZAÇÃO 3: Contador de frames para skip frames
        self._frame_counter = 0
        
        # OTIMIZAÇÃO 8: Fila FindFace global compartilhada (não cria worker próprio)
        self._findface_queue = findface_queue
        if self._findface_queue is not None:
            self.logger.info("Usando fila FindFace global compartilhada")
        elif self.findface_adapter is not None:
            self.logger.info("Fila FindFace não fornecida - modo síncrono")
        
        # OTIMIZAÇÃO 9: Fila assíncrona para inferência de landmarks em lote
        self._landmarks_queue: Optional[Queue] = None
        self._landmarks_worker: Optional[Thread] = None
        self._landmarks_worker_running = False
        self._landmarks_results: Dict[int, Optional[np.ndarray]] = {}  # Cache de resultados: event_id -> landmarks
        self._landmarks_batch_size = batch  # Tamanho do batch para landmarks
        
        if self.landmarks_model is not None:
            # Tamanho da fila: 3x o batch size para buffer
            landmarks_queue_size = self._landmarks_batch_size * 3
            self._landmarks_queue = Queue(maxsize=landmarks_queue_size)
            self._landmarks_worker_running = True
            self._landmarks_worker = Thread(
                target=self._landmarks_batch_worker,
                name=f"Landmarks-Worker-{camera.camera_id.value()}",
                daemon=True
            )
            self._landmarks_worker.start()
            self.logger.info(
                f"Worker assíncrono de landmarks iniciado "
                f"(fila: {landmarks_queue_size}, batch: {self._landmarks_batch_size})"
            )

    # Worker FindFace agora é global (pool de workers em run.py)
    # Método _findface_sender_worker removido - workers globais processam fila compartilhada
    
    def _landmarks_batch_worker(self):
        """Worker thread dedicado para inferência de landmarks em lote.
        
        Acumula crops de faces em batch e processa todos de uma vez,
        maximizando performance da GPU/CPU.
        """
        self.logger.info("Landmarks worker iniciado")
        
        batch_buffer = []  # Buffer temporário: [(event_id, face_crop), ...]
        
        while self._landmarks_worker_running:
            try:
                # Tenta pegar primeiro item (bloqueia até 0.1s)
                try:
                    item = self._landmarks_queue.get(timeout=0.1)
                    
                    if item is None:  # Sinal de parada
                        self.logger.info("Landmarks worker recebeu sinal de parada")
                        break
                    
                    batch_buffer.append(item)
                except:
                    # Timeout - processa batch parcial se existir
                    if batch_buffer and not self._landmarks_worker_running:
                        break
                    if not batch_buffer:
                        continue
                
                # Acumula até completar batch (ou timeout)
                while len(batch_buffer) < self._landmarks_batch_size:
                    try:
                        item = self._landmarks_queue.get(timeout=0.01)  # Timeout curto
                        if item is None:
                            break
                        batch_buffer.append(item)
                    except:
                        break  # Timeout - processa o que tem
                
                if not batch_buffer:
                    continue
                
                # Processa batch
                try:
                    # Prepara crops para inferência em lote
                    event_ids = [item[0] for item in batch_buffer]
                    crops = [item[1] for item in batch_buffer]
                    
                    # ================================================================
                    # OTIMIZAÇÃO CRÍTICA: BATCH INFERENCE VERDADEIRO
                    # ANTES: Loop individual - N chamadas ao modelo (lento)
                    # AGORA: Uma única chamada com batch - aproveita paralelização GPU
                    # Ganho: 4-8x mais rápido dependendo do batch size
                    # ================================================================
                    try:
                        # Infere landmarks em LOTE (uma única chamada ao modelo)
                        batch_results = self.landmarks_model.predict_batch(
                            face_crops=crops,
                            conf=self.conf,
                            verbose=False  # Desabilita logs em batch
                        )
                        
                        # Armazena resultados no cache
                        for event_id, result in zip(event_ids, batch_results):
                            if result is not None:
                                landmarks_array, _ = result
                                self._landmarks_results[event_id] = landmarks_array
                            else:
                                self._landmarks_results[event_id] = None
                    
                    except Exception as e:
                        # Fallback: se batch falhar, tenta individual
                        self.logger.warning(f"Batch inference falhou, usando fallback individual: {e}")
                        for event_id, crop in zip(event_ids, crops):
                            try:
                                if crop.size > 0:
                                    landmarks_result = self.landmarks_model.predict(
                                        face_crop=crop,
                                        conf=self.conf,
                                        verbose=False
                                    )
                                    
                                    if landmarks_result is not None:
                                        landmarks_array, _ = landmarks_result
                                        self._landmarks_results[event_id] = landmarks_array
                                    else:
                                        self._landmarks_results[event_id] = None
                                else:
                                    self._landmarks_results[event_id] = None
                            except Exception:
                                self._landmarks_results[event_id] = None
                    
                    # Marca tarefas como concluídas
                    for _ in batch_buffer:
                        self._landmarks_queue.task_done()
                    
                    # OTIMIZAÇÃO: Log de batch removido para reduzir I/O
                    
                except Exception as e:
                    self.logger.error(f"Erro no processamento de batch de landmarks: {e}")
                    # Marca tarefas como concluídas mesmo com erro
                    for _ in batch_buffer:
                        try:
                            self._landmarks_queue.task_done()
                        except:
                            pass
                
                # Limpa buffer
                batch_buffer.clear()
                
            except Exception as e:
                if not self._landmarks_worker_running:
                    break
                self.logger.error(f"Erro no landmarks worker: {e}")
                continue
        
        self.logger.info("Landmarks worker finalizado")
    
    def start(self):
        """Inicia o processamento do stream de vídeo"""
        self.running = True
        self.logger.info(
            f"ByteTrackDetectorService iniciado para câmera "
            f"{self.camera.camera_name.value()} (ID: {self.camera.camera_id.value()})"
        )
        self._process_stream()

    def stop(self):
        """Para o processamento do stream"""
        self.running = False
        
        # Finaliza serviço de salvamento de imagens
        if self.image_save_service is not None:
            try:
                self.image_save_service.stop()
                self.logger.info("ImageSaveService finalizado com sucesso")
            except Exception as e:
                self.logger.error(f"Erro ao finalizar ImageSaveService: {e}")
        
        # Finaliza worker de landmarks graciosamente
        if self._landmarks_queue is not None and self._landmarks_worker is not None:
            try:
                self._landmarks_worker_running = False
                
                # Envia sinal de parada
                try:
                    self._landmarks_queue.put(None, timeout=0.5)
                except:
                    self.logger.warning("Não foi possível enviar sinal de parada para landmarks worker")
                
                # Aguarda worker finalizar
                self._landmarks_worker.join(timeout=2.0)
                
                if self._landmarks_worker.is_alive():
                    self.logger.warning("Landmarks worker não finalizou no tempo esperado")
                else:
                    self.logger.info("Landmarks worker finalizado com sucesso")
            except Exception as e:
                self.logger.error(f"Erro ao finalizar landmarks worker: {e}")
        
        # NOTA: Workers FindFace são globais (gerenciados em run.py) - não para aqui
        
        self.logger.info(
            f"ByteTrackDetectorService finalizado para câmera "
            f"{self.camera.camera_name.value()}"
        )

    def _process_stream(self):
        """Processa o stream de vídeo frame a frame"""
        while self.running:
            try:
                for result in self.model.track(
                    source=self.camera.source.value(),
                    tracker=self.tracker,
                    persist=True,
                    conf=self.conf,
                    iou=self.iou,
                    show=self.show,
                    stream=self.stream,
                    batch=self.batch,
                    verbose=False,
                    imgsz=self.inference_size
                ):
                    if not self.running:
                        break
                    
                    # Incrementa contador de frames
                    self._frame_counter += 1
                    
                    # OTIMIZAÇÃO 3: Detectar apenas a cada N frames (tracking continua)
                    should_detect = (self._frame_counter % self.detection_skip_frames) == 0
                    
                    # Cria entidade Frame
                    frame_entity = self._create_frame_entity(result.orig_img)
                    current_frame_tracks = set()
                    
                    # Processa detecções do frame atual (se for o frame de detecção)
                    if should_detect and result.boxes is not None and result.boxes.id is not None:
                        for i, box in enumerate(result.boxes):
                            track_id = int(box.id[0])
                            
                            # FILTRO DE DETECÇÃO: Aplica filtros ANTES de criar evento
                            # Extrai dados para validação
                            confidence = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            bbox_width = x2 - x1
                            
                            # Filtra por confiança mínima
                            if confidence < self.min_confidence_threshold:
                                if self.verbose_log:
                                    self.logger.warning(
                                        f"Detecção rejeitada (Track {track_id}): "
                                        f"confiança insuficiente ({confidence:.4f} < {self.min_confidence_threshold:.4f})"
                                    )
                                continue
                            
                            # Filtra por largura mínima do bbox
                            if bbox_width < self.min_bbox_width:
                                if self.verbose_log:
                                    self.logger.warning(
                                        f"Detecção rejeitada (Track {track_id}): "
                                        f"bbox pequeno ({bbox_width}px < {self.min_bbox_width}px)"
                                    )
                                continue
                            
                            current_frame_tracks.add(track_id)
                            
                            # Cria Event para esta detecção
                            event = self._create_event_from_detection(
                                frame_entity, box, result.keypoints, i
                            )
                            
                            # Adiciona evento ao track
                            if track_id not in self.active_tracks:
                                self.active_tracks[track_id] = Track(
                                    id=IdVO(track_id),
                                    min_movement_percentage=self.min_movement_percentage
                                )
                                self.track_frame_count[track_id] = 0  # NOVO: inicializa contador
                            
                            self.active_tracks[track_id].add_event(
                                event,
                                min_threshold_pixels=self.min_movement_threshold
                            )
                            self.track_frames_lost[track_id] = 0
                            self.track_frame_count[track_id] += 1  # NOVO: incrementa contador de frames
                            
                            # ATUALIZADO: Verifica se o track atingiu o limite de FRAMES
                            if self.track_frame_count[track_id] >= self.max_frames_per_track:
                                self.logger.info(
                                    f"Track {track_id} atingiu o limite de {self.max_frames_per_track} frames. "
                                    "Finalizando track."
                                )
                                self._finalize_track(track_id)
                                current_frame_tracks.discard(track_id)
                                continue
                            
                            # OTIMIZAÇÃO: Log verbose removido do loop principal para evitar gargalo de I/O
                            # Use logger.debug apenas quando absolutamente necessário para debugging
                    
                    # Atualiza tracks perdidos
                    self._update_lost_tracks(current_frame_tracks)

            except KeyboardInterrupt:
                self.logger.info("Execução interrompida pelo usuário.")
                break
            except Exception as e:
                self.logger.exception(f"Erro no stream RTSP: {e}. Tentando reconectar em 5 segundos...")
                time.sleep(5)
        
        # Finaliza todos os tracks restantes ao encerrar
        self._finalize_all_tracks()

    def _create_frame_entity(self, frame_array: np.ndarray) -> Frame:
        """
        Cria uma entidade Frame a partir de um numpy array.
        OTIMIZAÇÃO: Usa FullFrameVO sem cópia (copy=False) - economiza ~70% memória.
        
        :param frame_array: Array numpy do frame.
        :return: Entidade Frame.
        """
        self._frame_id_counter += 1
        return Frame(
            id=IdVO(self._frame_id_counter),
            full_frame=FullFrameVO(frame_array, copy=False),  # OTIMIZAÇÃO 4: Sem cópia
            camera_id=self.camera.camera_id,
            camera_name=self.camera.camera_name,
            camera_token=self.camera.camera_token,
            timestamp=TimestampVO.now()
        )

    def _create_event_from_detection(
        self,
        frame: Frame,
        box,
        keypoints,
        index: int
    ) -> Event:
        """
        Cria uma entidade Event a partir de uma detecção YOLO.
        
        :param frame: Entidade Frame onde a detecção ocorreu.
        :param box: Box da detecção YOLO.
        :param keypoints: Keypoints da detecção.
        :param index: Índice da detecção no frame.
        :return: Entidade Event.
        """
        self._event_id_counter += 1
        event_id = self._event_id_counter
        
        # Extrai bbox
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox = BboxVO((x1, y1, x2, y2))
        
        # Extrai confiança
        confidence = ConfidenceVO(float(box.conf[0]))
        
        # Enfileira crop para inferência assíncrona de landmarks (se disponível)
        landmarks_array = None
        if self.landmarks_model is not None and self._landmarks_queue is not None:
            try:
                # Obtém o ndarray do frame e faz o crop usando coordenadas do bbox
                frame_array = frame.full_frame.ndarray_readonly
                face_crop = frame_array[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
                
                # Enfileira para processamento assíncrono em lote
                if face_crop.size > 0:
                    try:
                        self._landmarks_queue.put_nowait((event_id, face_crop.copy()))
                    except:
                        # Fila cheia - usa fallback (sem log para evitar spam)
                        pass
                
                # Tenta buscar resultado já processado (se existir)
                landmarks_array = self._landmarks_results.pop(event_id, None)
                    
            except Exception:
                # Erro ao processar landmarks - usa fallback silenciosamente
                pass
        
        # Fallback: usa landmarks do modelo de detecção (se disponível)
        if landmarks_array is None and keypoints is not None and len(keypoints) > index:
            kpts = keypoints[index].xy[0].cpu().numpy()
            landmarks_array = kpts
        
        landmarks = LandmarksVO(landmarks_array)
        
        # Cria evento (o face_quality_score é calculado automaticamente)
        event = Event(
            id=IdVO(event_id),
            frame=frame,
            bbox=bbox,
            confidence=confidence,
            landmarks=landmarks
        )
        
        return event

    def is_valid(self, track: Track) -> tuple[bool, str]:
        """
        Valida se um track atende às condições necessárias para ser considerado válido.
        
        Um track é válido se possui movimento significativo.
        
        NOTA: Filtros de confiança e tamanho do bbox são aplicados durante a detecção,
        antes de criar os eventos. Portanto, todos os eventos do track já passaram
        por esses filtros.
        
        :param track: Track a ser validado.
        :return: Tupla (is_valid, reason) onde:
                 - is_valid: True se o track for válido, False caso contrário
                 - reason: String vazia se válido, ou descrição do motivo da invalidação
        """
        # Verifica se possui melhor evento
        best_event = track.get_best_event()
        if best_event is None:
            return (False, "sem melhor evento")
        
        # Verifica movimento
        if not track.has_movement:
            return (False, "sem movimento significativo")
        
        return (True, "")

    def _update_lost_tracks(self, current_frame_tracks: set):
        """
        Atualiza contadores de frames perdidos e finaliza tracks.
        
        :param current_frame_tracks: Set com IDs dos tracks presentes no frame atual.
        """
        tracks_to_finalize = []
        
        for track_id in list(self.track_frames_lost.keys()):
            if track_id not in current_frame_tracks:
                self.track_frames_lost[track_id] += 1
                
                if self.track_frames_lost[track_id] >= self.max_frames_lost:
                    tracks_to_finalize.append(track_id)
        
        # Finaliza tracks perdidos
        for track_id in tracks_to_finalize:
            self._finalize_track(track_id)

    def _finalize_track(self, track_id: int):
        """
        Finaliza um track: encontra melhor evento, salva e envia para FindFace.
        
        :param track_id: ID do track a ser finalizado.
        """
        if track_id not in self.active_tracks:
            return
        
        track = self.active_tracks[track_id]
        
        if track.is_empty:
            self.logger.warning(f"Track {track_id} vazio, não será processado")
            del self.active_tracks[track_id]
            del self.track_frames_lost[track_id]
            if track_id in self.track_frame_count:  # NOVO: limpa contador de frames
                del self.track_frame_count[track_id]
            return
        
        # Verifica se o track é válido
        is_valid, invalid_reason = self.is_valid(track)
        
        # Obtém estatísticas de movimento
        has_movement = track.has_movement
        movement_stats = track.get_movement_statistics()
        
        # Obtém melhor evento do track
        best_event = track.get_best_event()
        best_confidence = best_event.confidence.value() if best_event else 0.0
        
        # ATUALIZADO: Log com informação de frames processados
        total_frames = self.track_frame_count.get(track_id, 0)
        
        self.logger.info(
            f"Track {track_id} finalizado após {self.track_frames_lost[track_id]} frames perdidos. "
            f"Total de frames: {total_frames} | "  # ATUALIZADO
            f"Total de eventos: {track.event_count} | "
            f"Movimento detectado: {has_movement} | "
            f"Distância média: {movement_stats['average_distance']:.2f}px | "
            f"Distância máxima: {movement_stats['max_distance']:.2f}px | "
            f"Confiança: {best_confidence:.2f} | "
            f"Válido: {is_valid}"
        )
        
        # Remove track da memória ANTES de processar
        del self.active_tracks[track_id]
        del self.track_frames_lost[track_id]
        if track_id in self.track_frame_count:  # NOVO: limpa contador de frames
            del self.track_frame_count[track_id]
        
        # OTIMIZAÇÃO 7: Coleta de lixo periódica a cada 500 tracks (reduz overhead)
        self._tracks_finalized_count += 1
        if self._tracks_finalized_count % 500 == 0:
            import gc
            gc.collect()
            self.logger.debug(f"Coleta de lixo executada após {self._tracks_finalized_count} tracks finalizados")
        
        if best_event is None:
            self.logger.warning(f"Track {track_id} não possui melhor evento")
            return
        
        # Salva melhor face (sempre salva, mas cor do bbox depende da validade)
        self._save_best_event(track_id, best_event, track.event_count, has_movement, is_valid)
        
        # Envia para FindFace apenas se o track for válido
        if self.findface_adapter is not None and is_valid:
            self._send_best_event_to_findface(track_id, best_event, track.event_count)
        elif not is_valid:
            # Log detalhado do motivo da invalidação
            self.logger.warning(
                f"Track {track_id} INVÁLIDO - Descartado | "
                f"Razão: {invalid_reason} | "
                f"Confiança: {best_confidence:.4f} | "
                f"Largura bbox: {best_event.bbox.width}px"
            )

    def _save_best_event(self, track_id: int, event: Event, total_events: int, has_movement: bool, is_valid: bool):
        """
        Salva o melhor evento do track em disco com bbox desenhado.
        
        :param track_id: ID do track.
        :param event: Melhor evento do track.
        :param total_events: Total de eventos no track.
        :param has_movement: Se o track teve movimento significativo.
        :param is_valid: Se o track é válido para envio ao FindFace.
        """
        try:
            # OTIMIZAÇÃO 5: Usa ndarray_readonly + copia apenas uma vez
            frame_with_bbox = event.frame.full_frame.ndarray_readonly.copy()
            
            x1, y1, x2, y2 = event.bbox.value()
            
            # Cor do bbox baseada na validade:
            # - Vermelho: inválido
            # - Verde: válido com movimento
            # - Amarelo: válido sem movimento (caso não usado, mantido para consistência)
            if not is_valid:
                bbox_color = (0, 0, 255)  # Vermelho para inválidos
                status_label = "INVALID"
            elif has_movement:
                bbox_color = (0, 255, 0)  # Verde para válidos com movimento
                status_label = "VALID"
            else:
                bbox_color = (0, 255, 255)  # Amarelo para válidos sem movimento
                status_label = "STATIC"
            
            # Desenha bbox
            cv2.rectangle(frame_with_bbox, (x1, y1), (x2, y2), bbox_color, 2)
            
            # Label
            label = (
                f"Track {track_id} | "
                f"{status_label} | "
                f"Quality: {event.face_quality_score.value():.4f} | "
                f"Conf: {event.confidence.value():.2f}"
            )
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                frame_with_bbox,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                bbox_color,
                -1
            )
            cv2.putText(
                frame_with_bbox,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # Texto branco para melhor contraste
                2
            )
            
            # Nome do arquivo
            timestamp_str = event.frame.timestamp.value().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            # Prefixo baseado na validade
            if not is_valid:
                prefix = "INVALID"
            elif has_movement:
                prefix = "VALID"
            else:
                prefix = "STATIC"
            
            # Sanitiza o nome da câmera para usar no filename (remove caracteres inválidos)
            camera_name_clean = self.camera.camera_name.value().replace(" ", "-").replace("/", "-").replace("\\", "-")
            camera_id = self.camera.camera_id.value()
            
            filename = f"{prefix}_Camera-{camera_id}-{camera_name_clean}_Track_{track_id}_{timestamp_str}.jpg"
            
            # Salva no disco apenas se habilitado
            if self.save_images:
                from pathlib import Path
                filepath = Path(self.project_dir) / self.results_dir / filename
                filepath.parent.mkdir(parents=True, exist_ok=True)
                
                # OTIMIZAÇÃO: Salvamento assíncrono via ImageSaveService
                if self.image_save_service is not None:
                    self.image_save_service.save_async(frame_with_bbox, filepath, jpeg_quality=95)
                else:
                    # Fallback síncrono se o serviço não foi fornecido
                    cv2.imwrite(str(filepath), frame_with_bbox, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # Log de salvamento apenas em modo verboso
                if self.verbose_log:
                    status_msg = "VÁLIDO" if is_valid else "INVÁLIDO"
                    self.logger.info(
                        f"Melhor face salva ({status_msg}): {filename} "
                        f"(quality={event.face_quality_score.value():.4f}, "
                        f"conf={event.confidence.value():.2f}) | "
                        f"Total de eventos: {total_events}"
                    )
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar evento do track {track_id}: {e}", exc_info=True)

    def _send_best_event_to_findface(self, track_id: int, event: Event, total_events: int):
        """
        Enfileira o melhor evento do track para envio assíncrono ao FindFace.
        OTIMIZAÇÃO 8: Usa fila global compartilhada processada por pool de workers.
        
        :param track_id: ID do track.
        :param event: Melhor evento do track.
        :param total_events: Total de eventos no track.
        """
        if self.findface_adapter is None:
            self.logger.warning(f"FindFace adapter não configurado. Track {track_id} não será enviado.")
            return
        
        if self._findface_queue is None:
            self.logger.error(f"Fila FindFace não inicializada. Track {track_id} não será enviado.")
            return
        
        try:
            # Enfileira evento para processamento assíncrono (non-blocking)
            # Formato: (camera_id, camera_name, track_id, event, total_events)
            event_data = (
                self.camera.camera_id.value(),
                self.camera.camera_name.value(),
                track_id,
                event,
                total_events
            )
            self._findface_queue.put_nowait(event_data)
            
            if self.verbose_log:
                self.logger.debug(
                    f"Track {track_id} enfileirado para envio FindFace "
                    f"(fila global: {self._findface_queue.qsize()}/{self._findface_queue.maxsize})"
                )
            
        except Exception as e:
            # Fila cheia - descarta evento e loga warning
            self.logger.warning(
                f"⚠ FindFace - Fila CHEIA - Track {track_id} DESCARTADO: "
                f"quality={event.face_quality_score.value():.4f}, "
                f"total_events={total_events}, "
                f"fila_size={self._findface_queue.qsize()}, "
                f"erro={e}"
            )

    def _finalize_all_tracks(self):
        """Finaliza todos os tracks ativos"""
        track_ids = list(self.active_tracks.keys())
        for track_id in track_ids:
            self._finalize_track(track_id)
        self.logger.info("Todos os tracks foram finalizados")
