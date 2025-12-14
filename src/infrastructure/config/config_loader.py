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
    ByteTrackConfig,
    ProcessingConfig,
    StorageConfig,
    CameraConfig,
    DetectionFilterConfig,
    FaceQualityConfig,
    MovementConfig,
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
        findface_config.camera_prefix = yaml_config.get("prefixo_grupo_camera_findface", "EXTERNO")
        
        # Monta configurações
        yolo_config = YOLOConfig(
            model_path=yaml_config.get("face_detection_model", "yolov8n-face.pt"),
            landmarks_model_path=yaml_config.get("landmarks_detection_model", "yolov8n-face.pt"),
            conf_threshold=yaml_config.get("conf", 0.1),
            iou_threshold=yaml_config.get("iou", 0.2)
        )
        
        bytetrack_config = ByteTrackConfig(
            tracker_config=yaml_config.get("tracker", "bytetrack.yaml"),
            max_frames_lost=yaml_config.get("max_frames_lost", 30),
            max_frames_per_track=yaml_config.get("max_frames_por_track", 900)
        )
        
        # Carrega gpu_devices do YAML (pode ser lista ou int único)
        gpu_devices_value = yaml_config.get("gpu_devices", yaml_config.get("gpu_index", 0))
        if isinstance(gpu_devices_value, int):
            gpu_devices = [gpu_devices_value]
        elif isinstance(gpu_devices_value, list):
            gpu_devices = gpu_devices_value
        else:
            gpu_devices = [0]  # Fallback para GPU 0
        
        processing_config = ProcessingConfig(
            gpu_devices=gpu_devices,
            show_video=yaml_config.get("show", True),
            verbose_log=yaml_config.get("verbose_log", False)
        )
        
        storage_config = StorageConfig(
            save_images=yaml_config.get("salvamento_imagens", {}).get("habilitado", True),
            project_dir=yaml_config.get("salvamento_imagens", {}).get("project", yaml_config.get("project", "./imagens/")),
            results_dir=yaml_config.get("salvamento_imagens", {}).get("name", yaml_config.get("name", "rtsp_byte_track_results"))
        )
        
        movement_config = MovementConfig(
            min_movement_threshold_pixels=yaml_config.get("movimento", {}).get("limiar_minimo_pixels", 5.0),
            min_movement_frame_percentage=yaml_config.get("movimento", {}).get("percentual_minimo_frames", 0.1)
        )
        
        detection_filter_config = DetectionFilterConfig(
            min_confidence=yaml_config.get("filtro_deteccao", {}).get("confianca_minima", 0.45),
            min_bbox_width=yaml_config.get("filtro_deteccao", {}).get("largura_minima_bbox", 60)
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
            detection_skip_frames=yaml_config.get("performance", {}).get("detection_skip_frames", 1),
            findface_queue_size=yaml_config.get("performance", {}).get("findface_queue_size", 200),
            jpeg_compression=yaml_config.get("performance", {}).get("jpeg_compression", 95),
            gpu_batch_size=yaml_config.get("performance", {}).get("gpu_batch_size", 32),
            cpu_batch_size=yaml_config.get("performance", {}).get("cpu_batch_size", 1)
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
            bytetrack=bytetrack_config,
            processing=processing_config,
            storage=storage_config,
            movement=movement_config,
            detection_filter=detection_filter_config,
            face_quality=face_quality_config,
            tensorrt=tensorrt_config,
            openvino=openvino_config,
            performance=performance_config,
            cameras=cameras
        )