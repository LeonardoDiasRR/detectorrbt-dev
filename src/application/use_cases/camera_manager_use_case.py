"""
Gerenciador dinâmico de câmeras (Application Layer).
Monitora status das câmeras no FindFace e gerencia threads de processamento.
"""

import logging
import threading
import time
from typing import Dict, Optional, List, Any
from queue import Queue

from src.domain.entities import Camera
from src.domain.repositories import CameraRepository
from src.domain.interfaces import IFaceDetector, ILandmarksDetector
from src.domain.services import (
    EventCreationService,
    MovementDetectionService,
    TrackValidationService,
    TrackLifecycleService,
    FaceQualityService,
    ImageSaveService
)
from src.domain.adapters import FindfaceAdapter
from src.application.use_cases import ProcessFaceDetectionUseCase
from src.infrastructure.config.settings import AppSettings


class CameraManager:
    """
    Gerenciador centralizado de câmeras com monitoramento dinâmico.
    Responsabilidades:
    - Verificar periodicamente status das câmeras no FindFace
    - Iniciar threads para câmeras ativas sem processamento
    - Finalizar threads de câmeras inativas
    - Monitorar saúde das câmeras (taxa de sucesso FindFace)
    """

    def __init__(
        self,
        camera_repository: CameraRepository,
        findface_queue: Queue,
        findface_adapter: FindfaceAdapter,
        settings: AppSettings,
        # Serviços compartilhados
        event_creation_service: EventCreationService,
        movement_detection_service: MovementDetectionService,
        track_validation_service: TrackValidationService,
        track_lifecycle_service: TrackLifecycleService,
        face_quality_service: Optional[FaceQualityService] = None,
        # Factory functions para criar modelos
        model_factory: Optional[Any] = None,
        face_detector_factory: Optional[Any] = None,
        landmarks_detector_factory: Optional[Any] = None,
        image_save_service_factory: Optional[Any] = None,
        # GPU devices disponíveis
        gpu_devices: Optional[List[int]] = None
    ):
        """
        Inicializa o gerenciador de câmeras.
        
        :param camera_repository: Repositório para buscar câmeras.
        :param findface_queue: Fila compartilhada para envio ao FindFace.
        :param findface_adapter: Adapter para comunicação com FindFace.
        :param settings: Configurações da aplicação.
        :param event_creation_service: Serviço compartilhado de criação de eventos.
        :param movement_detection_service: Serviço compartilhado de detecção de movimento.
        :param track_validation_service: Serviço compartilhado de validação de tracks.
        :param track_lifecycle_service: Serviço compartilhado de ciclo de vida de tracks.
        :param face_quality_service: Serviço opcional de qualidade facial.
        :param model_factory: Factory para criar modelos de detecção.
        :param face_detector_factory: Factory para criar face detectors.
        :param landmarks_detector_factory: Factory para criar landmarks detectors.
        :param image_save_service_factory: Factory para criar image save services.
        :param gpu_devices: Lista de IDs de GPUs disponíveis.
        """
        self.camera_repository = camera_repository
        self.findface_queue = findface_queue
        self.findface_adapter = findface_adapter
        self.settings = settings
        
        # Serviços compartilhados
        self.event_creation_service = event_creation_service
        self.movement_detection_service = movement_detection_service
        self.track_validation_service = track_validation_service
        self.track_lifecycle_service = track_lifecycle_service
        self.face_quality_service = face_quality_service
        
        # Factories
        self.model_factory = model_factory
        self.face_detector_factory = face_detector_factory
        self.landmarks_detector_factory = landmarks_detector_factory
        self.image_save_service_factory = image_save_service_factory
        
        # GPUs disponíveis
        self.gpu_devices = gpu_devices if gpu_devices else [0]
        
        # Estado do gerenciamento
        self.active_processors: Dict[int, ProcessFaceDetectionUseCase] = {}
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_running = False
        
        # Lock para sincronização de criação de models (TensorRT thread-safety)
        self.model_creation_lock = threading.Lock()
        
        # Contador de câmeras por GPU (para balanceamento)
        self.gpu_camera_count: Dict[int, int] = {gpu_id: 0 for gpu_id in self.gpu_devices}
        
        self.logger = logging.getLogger(self.__class__.__name__)

    def start_monitoring(self) -> None:
        """
        Inicia o monitoramento dinâmico de câmeras.
        Cria thread que verifica periodicamente o status das câmeras.
        """
        if not self.settings.camera_monitoring.enabled:
            self.logger.info("Monitoramento de câmeras DESABILITADO por configuração")
            return
        
        self.monitoring_running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_loop,
            name="CameraMonitor",
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info(
            f"Monitoramento de câmeras INICIADO "
            f"(intervalo: {self.settings.camera_monitoring.check_interval}s)"
        )

    def stop_monitoring(self) -> None:
        """
        Para o monitoramento e finaliza todas as câmeras ativas.
        """
        self.logger.info("Parando monitoramento de câmeras...")
        self.monitoring_running = False
        
        # Aguarda thread de monitoramento finalizar
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
        
        # Para todos os processors ativos
        camera_ids = list(self.active_processors.keys())
        for camera_id in camera_ids:
            self._stop_camera(camera_id)
        
        self.logger.info("Monitoramento de câmeras FINALIZADO")

    def _monitor_loop(self) -> None:
        """
        Loop principal de monitoramento (executa em thread separada).
        Verifica periodicamente o status das câmeras e atualiza processamento.
        """
        while self.monitoring_running:
            try:
                # 1. Obtém lista completa de câmeras do FindFace
                all_cameras = self.camera_repository.get_cameras()
                
                # 2. Separa câmeras ativas e inativas
                active_cameras = {cam.camera_id.value(): cam for cam in all_cameras if cam.active}
                inactive_camera_ids = {cam.camera_id.value() for cam in all_cameras if not cam.active}
                
                # 3. Para câmeras inativas que estão rodando
                for camera_id in inactive_camera_ids:
                    if camera_id in self.active_processors:
                        self.logger.info(
                            f"Câmera {camera_id} ficou INATIVA - finalizando processamento"
                        )
                        self._stop_camera(camera_id)
                
                # 4. Inicia câmeras ativas que não estão rodando
                for camera_id, camera in active_cameras.items():
                    if camera_id not in self.active_processors:
                        self.logger.info(
                            f"Nova câmera ATIVA detectada: {camera.camera_name.value()} "
                            f"(ID: {camera_id}) - iniciando processamento em thread separada"
                        )
                        # Inicia câmera em thread separada para não bloquear o monitoramento
                        start_thread = threading.Thread(
                            target=self._start_camera,
                            args=(camera,),
                            name=f"StartCamera-{camera_id}",
                            daemon=True
                        )
                        start_thread.start()
                
                # 5. Verifica saúde das câmeras ativas
                self._check_cameras_health()
                
            except Exception as e:
                self.logger.error(f"Erro no loop de monitoramento: {e}", exc_info=True)
            
            # Aguarda intervalo configurado
            time.sleep(self.settings.camera_monitoring.check_interval)

    def _start_camera(self, camera: Camera) -> None:
        """
        Inicia processamento de uma câmera.
        Cria model, detector, processor e thread.
        
        :param camera: Entidade Camera a iniciar.
        """
        camera_id = camera.camera_id.value()
        
        # Proteção contra race condition - verifica se câmera já está ativa
        if camera_id in self.active_processors:
            self.logger.debug(f"Câmera {camera_id} já está ativa - ignorando duplicação")
            return
        
        try:
            # Seleciona GPU com menos câmeras (balanceamento)
            gpu_id = min(self.gpu_camera_count, key=self.gpu_camera_count.get)  # type: ignore
            
            # CRITICAL: Lock para criação sequencial de models (TensorRT thread-safety)
            with self.model_creation_lock:
                # Cria model de detecção
                if self.model_factory:
                    detection_model = self.model_factory(gpu_id=gpu_id)
                else:
                    self.logger.error("model_factory não configurado - impossível criar model")
                    return
                
                # Cria face detector
                if self.face_detector_factory:
                    face_detector = self.face_detector_factory(detection_model)
                else:
                    self.logger.error("face_detector_factory não configurado")
                    return
            
            # Cria landmarks detector (opcional)
            landmarks_detector = None
            if self.landmarks_detector_factory:
                landmarks_detector = self.landmarks_detector_factory(gpu_id=gpu_id)
            
            # Cria image save service (opcional)
            image_save_service = None
            if self.image_save_service_factory and self.settings.storage.save_images:
                image_save_service = self.image_save_service_factory(camera_name=camera.camera_name.value())
            
            # Cria processor
            processor = ProcessFaceDetectionUseCase(
                camera=camera,
                face_detector=face_detector,
                findface_adapter=self.findface_adapter,
                findface_queue=self.findface_queue,
                event_creation_service=self.event_creation_service,
                movement_detection_service=self.movement_detection_service,
                track_validation_service=self.track_validation_service,
                track_lifecycle_service=self.track_lifecycle_service,
                landmarks_detector=landmarks_detector,
                gpu_id=gpu_id,
                image_save_service=image_save_service,
                face_quality_service=self.face_quality_service,
                tracker_config=self.settings.yolo.tracker,
                show_video=self.settings.display.exibir_na_tela,
                conf_threshold=self.settings.yolo.confidence_threshold,
                iou_threshold=self.settings.yolo.iou_threshold,
                landmark_conf_threshold=self.settings.landmark.confidence_threshold,
                landmark_iou_threshold=self.settings.landmark.iou_threshold,
                max_frames_lost=self.settings.bytetrack.max_age,
                save_images=self.settings.storage.save_images,
                project_dir=self.settings.storage.project_dir,
                results_dir=self.settings.storage.results_dir,
                min_movement_threshold=self.settings.track.min_movement_pixels,
                min_movement_percentage=self.settings.track.min_movement_percentage,
                min_confidence_threshold=self.settings.filter.min_confidence,
                min_bbox_area=self.settings.filter.min_bbox_area,
                max_frames_per_track=self.settings.bytetrack.max_frames,
                inference_size=(self.settings.performance.inference_size, self.settings.performance.inference_size),
                detection_skip_frames=self.settings.performance.detection_skip_frames,
                jpeg_quality=self.settings.compression.jpeg_quality,
                success_tracking_window=self.settings.camera_monitoring.success_tracking_window
            )
            
            # Cria e inicia thread
            thread = threading.Thread(
                target=processor.start,
                name=f"Camera-{camera_id}-{camera.camera_name.value()}",
                daemon=True
            )
            processor.thread = thread
            thread.start()
            
            # Registra processor ativo
            self.active_processors[camera_id] = processor
            self.gpu_camera_count[gpu_id] += 1
            
            self.logger.info(
                f"✓ Câmera {camera.camera_name.value()} (ID: {camera_id}) iniciada na GPU {gpu_id}"
            )
            
        except Exception as e:
            self.logger.error(
                f"✗ Erro ao iniciar câmera {camera.camera_name.value()} (ID: {camera_id}): {e}",
                exc_info=True
            )

    def _stop_camera(self, camera_id: int) -> None:
        """
        Finaliza processamento de uma câmera.
        
        :param camera_id: ID da câmera a finalizar.
        """
        processor = self.active_processors.get(camera_id)
        if not processor:
            return
        
        try:
            # Para processor
            processor.stop()
            
            # Aguarda thread finalizar (timeout de 5s)
            if processor.thread and processor.thread.is_alive():
                processor.thread.join(timeout=5.0)
                
                if processor.thread.is_alive():
                    self.logger.warning(
                        f"Thread da câmera {camera_id} não finalizou no timeout - "
                        f"será deixada como daemon"
                    )
            
            # Atualiza contadores GPU
            self.gpu_camera_count[processor.gpu_id] -= 1
            
            # Remove do registro
            del self.active_processors[camera_id]
            
            self.logger.info(f"✓ Câmera {camera_id} finalizada com sucesso")
            
        except Exception as e:
            self.logger.error(f"✗ Erro ao finalizar câmera {camera_id}: {e}", exc_info=True)

    def _check_cameras_health(self) -> None:
        """
        Verifica saúde das câmeras ativas.
        Monitora taxa de sucesso de envio ao FindFace.
        """
        for camera_id, processor in list(self.active_processors.items()):
            try:
                success_rate = processor.get_success_rate()
                
                if success_rate < self.settings.camera_monitoring.min_success_rate:
                    self.logger.warning(
                        f"⚠ Câmera {camera_id} ({processor.camera.camera_name.value()}) com baixa taxa de sucesso: "
                        f"{success_rate:.1%} < {self.settings.camera_monitoring.min_success_rate:.1%}"
                    )
                
            except Exception as e:
                self.logger.error(f"Erro ao verificar saúde da câmera {camera_id}: {e}")

    def get_active_cameras_count(self) -> int:
        """Retorna quantidade de câmeras ativas."""
        return len(self.active_processors)

    def get_camera_stats(self, camera_id: int) -> Optional[Dict[str, Any]]:
        """
        Obtém estatísticas de uma câmera específica.
        
        :param camera_id: ID da câmera.
        :return: Dicionário com estatísticas ou None se não encontrada.
        """
        processor = self.active_processors.get(camera_id)
        if not processor:
            return None
        
        return {
            "camera_id": camera_id,
            "camera_name": processor.camera.camera_name.value(),
            "gpu_id": processor.gpu_id,
            "success_rate": processor.get_success_rate(),
            "frame_count": processor.frame_count,
            "active_tracks": len(processor.active_tracks),
            "thread_alive": processor.thread.is_alive() if processor.thread else False
        }
