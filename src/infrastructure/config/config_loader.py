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
    TensorRTConfig,
    OpenVINOConfig,
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
            persist=yaml_config.get("modelo_deteccao", {}).get("persist", False)
        )
        
        landmark_config = LandmarkConfig(
            model_path=yaml_config.get("modelo_landmark", {}).get("model_path", "yolov8n-face.pt"),
            confidence_threshold=yaml_config.get("modelo_landmark", {}).get("confidence_threshold", 0.1),
            iou_threshold=yaml_config.get("modelo_landmark", {}).get("iou_threshold", 0.75)
        )
        
        bytetrack_config = ByteTrackConfig(
            tracker_config=yaml_config.get("tracking", {}).get("tracker_config", "bytetrack.yaml"),
            max_age=yaml_config.get("tracking", {}).get("max_age", 30),
            max_frames=yaml_config.get("tracking", {}).get("max_frames", 900)
        )
        
        # Carrega gpu_devices do YAML (pode ser lista ou int único)
        gpu_devices_value = yaml_config.get("processing", {}).get("gpu_devices", [0])
        if isinstance(gpu_devices_value, int):
            gpu_devices = [gpu_devices_value]
        elif isinstance(gpu_devices_value, list):
            gpu_devices = gpu_devices_value
        else:
            gpu_devices = [0]  # Fallback para GPU 0
        
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
            min_bbox_width=yaml_config.get("filter", {}).get("min_bbox_width", 60)
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
        
        # Configuração TensorRT
        tensorrt_config = TensorRTConfig(
            enabled=yaml_config.get("tensorrt", {}).get("enabled", True),
            precision=yaml_config.get("tensorrt", {}).get("precision", "FP16"),
            workspace=yaml_config.get("tensorrt", {}).get("workspace", 4)
        )
        
        # Configuração OpenVINO
        openvino_config = OpenVINOConfig(
            enabled=yaml_config.get("openvino", {}).get("enabled", True),
            device=yaml_config.get("openvino", {}).get("device", "AUTO"),
            precision=yaml_config.get("openvino", {}).get("precision", "FP16")
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
            tensorrt=tensorrt_config,
            openvino=openvino_config,
            performance=performance_config,
            cameras=cameras
        )