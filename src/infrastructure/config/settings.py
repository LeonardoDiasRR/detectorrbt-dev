"""
Objeto de configuração centralizado.
Fornece acesso type-safe às configurações.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class CameraConfig:
    """Configuração de uma câmera individual."""
    id: int
    name: str
    url: str
    token: str = ""


@dataclass
class YOLOConfig:
    """Configuração do modelo YOLO para detecção de faces."""
    model_path: str = "yolov8n-face.pt"
    confidence_threshold: float = 0.1
    iou_threshold: float = 0.07
    tracker: str = "bytetrack.yaml"
    persist: bool = False
    backend: str = "pytorch"  # pytorch, tensorrt ou openvino
    precision: str = "FP16"   # FP32 ou FP16
    device: str = "cpu"       # cpu ou cuda:N (ex: cuda:0, cuda:0,1)
    cpu_batch_size: int = 1
    gpu_batch_size: int = 32
    
    def get_batch_size(self) -> int:
        """
        Retorna o tamanho do batch apropriado baseado no device configurado.
        
        :return: Tamanho do batch (cpu_batch_size se device='cpu', gpu_batch_size se cuda)
        """
        if self.device == "cpu":
            return self.cpu_batch_size
        else:
            # cuda:N, cuda:0,1, etc.
            return self.gpu_batch_size


@dataclass
class LandmarkConfig:
    """Configuração do modelo YOLO para detecção de landmarks."""
    model_path: str = "yolov8n-face.pt"
    confidence_threshold: float = 0.1
    iou_threshold: float = 0.75
    backend: str = "pytorch"  # pytorch, tensorrt ou openvino
    precision: str = "FP16"   # FP32 ou FP16
    device: str = "cpu"       # cpu ou cuda:N (ex: cuda:0, cuda:0,1)


@dataclass
class ByteTrackConfig:
    """Configuração do ByteTrack."""
    tracker_config: str = "bytetrack.yaml"
    max_age: int = 30
    max_frames: int = 900


@dataclass
class FindFaceConfig:
    """Configuração do FindFace."""
    url_base: str
    user: str
    password: str
    uuid: str
    group_prefix: str = "EXTERNO"


@dataclass
class DisplayConfig:
    """Configuração de exibição na tela."""
    exibir_na_tela: bool = True


@dataclass
class LoggingConfig:
    """Configuração de logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class StorageConfig:
    """Configuração de armazenamento."""
    save_images: bool = True
    project_dir: str = "./imagens/"
    results_dir: str = "rtsp_byte_track_results"


@dataclass
class CompressionConfig:
    """Configuração de compressão de imagens."""
    jpeg_quality: int = 95


@dataclass
class TrackConfig:
    """Configuração de detecção de movimento no track."""
    min_movement_pixels: float = 2.0
    min_movement_percentage: float = 0.05


@dataclass
class FilterConfig:
    """Configuração de filtros de detecção (aplicados antes de criar evento)."""
    min_confidence: float = 0.45
    min_bbox_area: int = 60


@dataclass
class QueuesConfig:
    """Configuração de filas assíncronas."""
    findface_queue_max_size: int = 200


@dataclass
class CameraMonitoringConfig:
    """Configuração de monitoramento dinâmico de câmeras."""
    enabled: bool = True
    check_interval: int = 10
    success_tracking_window: int = 100
    min_success_rate: float = 0.8


@dataclass
class FaceQualityConfig:
    """Configuração de pesos para cálculo de qualidade facial."""
    peso_confianca: float = 3.0
    peso_tamanho: float = 4.0
    peso_frontal: float = 6.0
    peso_proporcao: float = 1.0


@dataclass
class PerformanceConfig:
    """Configuração de otimizações de performance."""
    inference_size: int = 640
    detection_skip_frames: int = 1


@dataclass
class AppSettings:
    """
    Configurações completas da aplicação.
    Objeto imutável que centraliza todas as configurações.
    """
    findface: FindFaceConfig
    yolo: YOLOConfig
    landmark: LandmarkConfig
    bytetrack: ByteTrackConfig
    display: DisplayConfig
    logging: LoggingConfig
    storage: StorageConfig
    compression: CompressionConfig
    track: TrackConfig
    filter: FilterConfig
    queues: QueuesConfig
    camera_monitoring: CameraMonitoringConfig
    face_quality: FaceQualityConfig
    performance: PerformanceConfig
    cameras: List[CameraConfig]