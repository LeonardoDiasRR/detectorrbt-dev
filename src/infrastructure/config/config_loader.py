"""
Carregador de configurações.
Responsável por ler arquivos YAML e variáveis de ambiente,
convertendo-os em objetos de configuração type-safe.
"""

import os
import yaml
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


def get_available_device() -> str:
    """
    Detecta o device disponível no sistema.
    Se CUDA estiver disponível, retorna o valor configurado.
    Se CUDA não estiver disponível, retorna "cpu".
    
    :return: String do device ("0", "0,1", "cpu", etc).
    """
    try:
        import torch
        if torch.cuda.is_available():
            return None  # Sinaliza para usar a configuração do YAML
        else:
            return "cpu"
    except ImportError:
        # Se PyTorch não estiver disponível, usa CPU
        return "cpu"

from .settings import (
    AppSettings,
    FindFaceConfig,
    YOLOConfig,
    LandmarkConfig,
    ByteTrackConfig,
    ProcessingConfig,
    DisplayConfig,
    LoggingConfig,
    StorageConfig,
    CompressionConfig,
    TrackConfig,
    FilterConfig,
    QueuesConfig,
    CameraMonitoringConfig,
    CameraConfig,
    FaceQualityConfig,
    PerformanceConfig
)


class ConfigLoader:
    """Carrega configurações de arquivos e variáveis de ambiente."""
    
    @staticmethod
    def load_from_yaml(yaml_path: str = "config.yaml") -> dict:
        """
        Carrega configurações de arquivo YAML.
        
        :param yaml_path: Caminho para o arquivo YAML.
        :return: Dicionário com configurações.
        """
        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {yaml_path}")
        
        with open(yaml_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    @staticmethod
    def load_from_env() -> FindFaceConfig:
        """
        Carrega configurações do FindFace de variáveis de ambiente.
        
        :return: Configuração do FindFace.
        :raises ValueError: Se variáveis obrigatórias não estiverem definidas.
        """
        load_dotenv()
        
        required_vars = ["FINDFACE_URL", "FINDFACE_USER", "FINDFACE_PASSWORD", "FINDFACE_UUID"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Variáveis de ambiente obrigatórias não definidas: {', '.join(missing_vars)}")
        
        return FindFaceConfig(
            url_base=os.getenv("FINDFACE_URL", ""),
            user=os.getenv("FINDFACE_USER", ""),
            password=os.getenv("FINDFACE_PASSWORD", ""),
            uuid=os.getenv("FINDFACE_UUID", "")
        )
    
    @classmethod
    def load(cls, yaml_path: str = "config.yaml") -> AppSettings:
        """
        Carrega todas as configurações da aplicação.
        
        :param yaml_path: Caminho para o arquivo YAML.
        :return: Objeto AppSettings completo.
        """
        # Carrega do YAML
        yaml_config = cls.load_from_yaml(yaml_path)
        
        # Carrega FindFace do .env
        findface_config = cls.load_from_env()
        
        # Adiciona prefixo de câmera do YAML ao FindFace
        findface_config.group_prefix = yaml_config.get("findface", {}).get("group_prefix", "EXTERNO")
        
        # Monta configurações de modelos
        yolo_config = YOLOConfig(
            model_path=yaml_config.get("modelo_deteccao", {}).get("model_path", "yolov8n-face.pt"),
            confidence_threshold=yaml_config.get("modelo_deteccao", {}).get("confidence_threshold", 0.1),
            iou_threshold=yaml_config.get("modelo_deteccao", {}).get("iou_threshold", 0.07),
            tracker=yaml_config.get("modelo_deteccao", {}).get("tracker", "bytetrack.yaml"),
            persist=yaml_config.get("modelo_deteccao", {}).get("persist", False),
            backend=yaml_config.get("modelo_deteccao", {}).get("backend", "pytorch"),
            precision=yaml_config.get("modelo_deteccao", {}).get("precision", "FP16")
        )
        
        landmark_config = LandmarkConfig(
            model_path=yaml_config.get("modelo_landmark", {}).get("model_path", "yolov8n-face.pt"),
            confidence_threshold=yaml_config.get("modelo_landmark", {}).get("confidence_threshold", 0.1),
            iou_threshold=yaml_config.get("modelo_landmark", {}).get("iou_threshold", 0.75),
            backend=yaml_config.get("modelo_landmark", {}).get("backend", "pytorch"),
            precision=yaml_config.get("modelo_landmark", {}).get("precision", "FP16")
        )
        
        bytetrack_config = ByteTrackConfig(
            tracker_config=yaml_config.get("tracking", {}).get("tracker_config", "bytetrack.yaml"),
            max_age=yaml_config.get("tracking", {}).get("max_age", 30),
            max_frames=yaml_config.get("tracking", {}).get("max_frames", 900)
        )
        
        # Carrega gpu_devices do YAML (string separada por vírgula, ex: "0,1,2")
        gpu_devices_value = yaml_config.get("processing", {}).get("gpu_devices", "0")
        
        # Verifica disponibilidade de GPU e usa fallback para CPU se necessário
        available_device = get_available_device()
        if available_device == "cpu":
            # CUDA não disponível - força uso de CPU
            gpu_devices = "cpu"
        else:
            # CUDA disponível - mantém a configuração do YAML
            # Mantém como string - YOLO e PyTorch aceitam strings como "0", "0,1", "cuda:0"
            try:
                if isinstance(gpu_devices_value, str):
                    # Normaliza: remove espaços, mantém como string
                    gpu_devices = ','.join(x.strip() for x in gpu_devices_value.split(','))
                elif isinstance(gpu_devices_value, int):
                    # Compatibilidade: converte int para string
                    gpu_devices = str(gpu_devices_value)
                elif isinstance(gpu_devices_value, list):
                    # Compatibilidade: converte lista para string
                    gpu_devices = ','.join(str(x) for x in gpu_devices_value)
                else:
                    gpu_devices = "0"  # Fallback para GPU 0
            except (ValueError, AttributeError):
                gpu_devices = "0"  # Fallback para GPU 0 em caso de erro
        
        processing_config = ProcessingConfig(
            gpu_devices=gpu_devices,
            cpu_batch_size=yaml_config.get("processing", {}).get("cpu_batch_size", 1),
            gpu_batch_size=yaml_config.get("processing", {}).get("gpu_batch_size", 32)
        )
        
        display_config = DisplayConfig(
            exibir_na_tela=yaml_config.get("display", {}).get("exibir_na_tela", True)
        )
        
        logging_config = LoggingConfig(
            level=yaml_config.get("logging", {}).get("level", "INFO"),
            format=yaml_config.get("logging", {}).get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        
        storage_config = StorageConfig(
            save_images=yaml_config.get("storage", {}).get("save_images", True),
            project_dir=yaml_config.get("storage", {}).get("project_dir", "./imagens/"),
            results_dir=yaml_config.get("storage", {}).get("results_dir", "rtsp_byte_track_results")
        )
        
        compression_config = CompressionConfig(
            jpeg_quality=yaml_config.get("compression", {}).get("jpeg_quality", 95)
        )
        
        track_config = TrackConfig(
            min_movement_pixels=yaml_config.get("track", {}).get("min_movement_pixels", 50.0),
            min_movement_percentage=yaml_config.get("track", {}).get("min_movement_percentage", 0.1)
        )
        
        filter_config = FilterConfig(
            min_confidence=yaml_config.get("filter", {}).get("min_confidence", 0.45),
            min_bbox_area=yaml_config.get("filter", {}).get("min_bbox_area", 60)
        )
        
        queues_config = QueuesConfig(
            findface_queue_max_size=yaml_config.get("queues", {}).get("findface_queue_max_size", 200)
        )
        
        # Configuração de Monitoramento de Câmeras
        camera_monitoring_config = CameraMonitoringConfig(
            enabled=yaml_config.get("camera_monitoring", {}).get("enabled", True),
            check_interval=yaml_config.get("camera_monitoring", {}).get("check_interval", 10),
            success_tracking_window=yaml_config.get("camera_monitoring", {}).get("success_tracking_window", 100),
            min_success_rate=yaml_config.get("camera_monitoring", {}).get("min_success_rate", 0.8)
        )
        
        # Configuração de Qualidade Facial
        face_quality_config = FaceQualityConfig(
            peso_confianca=float(yaml_config.get("qualidade_face", {}).get("confianca_deteccao", 3.0)),
            peso_tamanho=float(yaml_config.get("qualidade_face", {}).get("tamanho_bbox", 4.0)),
            peso_frontal=float(yaml_config.get("qualidade_face", {}).get("face_frontal", 6.0)),
            peso_proporcao=float(yaml_config.get("qualidade_face", {}).get("proporcao_bbox", 1.0))
        )
        

        # Configuração de Performance
        performance_config = PerformanceConfig(
            inference_size=yaml_config.get("performance", {}).get("inference_size", 640),
            detection_skip_frames=yaml_config.get("performance", {}).get("detection_skip_frames", 1)
        )
        
        # Carrega câmeras do YAML
        cameras = [
            CameraConfig(
                id=cam.get("id", 0),
                name=cam.get("name", "Camera Local"),
                url=cam.get("url", ""),
                token=cam.get("token", "")
            )
            for cam in yaml_config.get("cameras", [])
        ]
        
        return AppSettings(
            findface=findface_config,
            yolo=yolo_config,
            landmark=landmark_config,
            bytetrack=bytetrack_config,
            processing=processing_config,
            display=display_config,
            logging=logging_config,
            storage=storage_config,
            compression=compression_config,
            track=track_config,
            filter=filter_config,
            queues=queues_config,
            camera_monitoring=camera_monitoring_config,
            face_quality=face_quality_config,
            performance=performance_config,
            cameras=cameras
        )