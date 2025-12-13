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
    """Configuração do modelo YOLO."""
    model_path: str = "yolov8n-face.pt"
    landmarks_model_path: str = "yolov8n-face.pt"
    conf_threshold: float = 0.1
    iou_threshold: float = 0.2


@dataclass
class ByteTrackConfig:
    """Configuração do ByteTrack."""
    tracker_config: str = "bytetrack.yaml"
    max_frames_lost: int = 30
    max_frames_per_track: int = 900


@dataclass
class FindFaceConfig:
    """Configuração do FindFace."""
    url_base: str
    user: str
    password: str
    uuid: str
    camera_prefix: str = "EXTERNO"


@dataclass
class ProcessingConfig:
    """Configuração de processamento."""
    gpu_devices: List[int] = None
    
    def __post_init__(self):
        """Inicializa valores padrão após criação."""
        if self.gpu_devices is None:
            self.gpu_devices = [0]
    show_video: bool = True
    verbose_log: bool = False


@dataclass
class StorageConfig:
    """Configuração de armazenamento."""
    save_images: bool = True
    project_dir: str = "./imagens/"
    results_dir: str = "rtsp_byte_track_results"


@dataclass
class MovementConfig:
    """Configuração de detecção de movimento."""
    min_movement_threshold_pixels: float = 50.0
    min_movement_frame_percentage: float = 0.1


@dataclass
class DetectionFilterConfig:
    """Configuração de filtros de detecção (aplicados antes de criar evento)."""
    min_confidence: float = 0.45
    min_bbox_width: int = 60


@dataclass
class FaceQualityConfig:
    """Configuração de pesos para cálculo de qualidade facial."""
    peso_confianca: float = 3.0
    peso_tamanho: float = 4.0
    peso_frontal: float = 6.0
    peso_proporcao: float = 1.0
    peso_nitidez: float = 1.0


@dataclass
class PerformanceConfig:
    """Configuração de otimizações de performance."""
    inference_size: int = 640
    detection_skip_frames: int = 1
    findface_queue_size: int = 200  # Tamanho da fila assíncrona FindFace


@dataclass
class TensorRTConfig:
    """Configuração do TensorRT."""
    enabled: bool = True
    precision: str = "FP16"  # FP16, FP32, INT8
    workspace: int = 4  # Workspace em GB


@dataclass
class OpenVINOConfig:
    """Configuração do OpenVINO."""
    enabled: bool = True
    device: str = "AUTO"  # AUTO, CPU, GPU, NPU
    precision: str = "FP16"  # FP16, FP32, INT8


@dataclass
class AppSettings:
    """
    Configurações completas da aplicação.
    Objeto imutável que centraliza todas as configurações.
    """
    findface: FindFaceConfig
    yolo: YOLOConfig
    bytetrack: ByteTrackConfig
    processing: ProcessingConfig
    storage: StorageConfig
    movement: MovementConfig
    detection_filter: DetectionFilterConfig
    face_quality: FaceQualityConfig
    performance: PerformanceConfig
    tensorrt: TensorRTConfig
    openvino: OpenVINOConfig
    cameras: List[CameraConfig]
    
    @property
    def device(self) -> str:
        """Retorna o dispositivo a ser usado (cuda ou cpu).
        
        Nota: Em configurações multi-GPU, retorna a primeira GPU da lista.
        O dispositivo específico é definido no momento da criação do modelo.
        """
        import torch
        if torch.cuda.is_available():
            return f"cuda:{self.processing.gpu_devices[0]}"
        return "cpu"