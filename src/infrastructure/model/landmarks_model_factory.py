# src/infrastructure/model/landmarks_model_factory.py
"""
Factory para criação de modelos de detecção de landmarks faciais.
Suporta diferentes backends (YOLO, TensorRT, OpenVINO).
"""

import logging
from pathlib import Path

from src.domain.services.landmarks_model_interface import ILandmarksModel
from src.infrastructure.model.landmarks_yolo_model_adapter import LandmarksYOLOModelAdapter

logger = logging.getLogger(__name__)


class LandmarksModelFactory:
    """
    Factory que cria instâncias de modelos de landmarks faciais
    baseado no tipo de arquivo e configuração.
    """
    
    @staticmethod
    def create(
        model_path: str,
        device: str = 'cpu',
        backend: str = 'pytorch',
        precision: str = 'FP16',
        image_size: int = 640
    ) -> ILandmarksModel:
        """
        Cria uma instância de ILandmarksModel baseado no modelo especificado.
        
        :param model_path: Caminho para o arquivo do modelo.
        :param device: Dispositivo para inferência ('cpu', 'cuda', etc).
        :param backend: Backend para usar ('pytorch' atualmente suportado).
        :param precision: Precisão do modelo ('FP16' ou 'FP32').
        :param image_size: Tamanho de imagem para inferência (640 ou 1280).
        :return: Instância de ILandmarksModel.
        :raises ValueError: Se o modelo não for suportado.
        :raises FileNotFoundError: Se o modelo não existir.
        """
        model_file = Path(model_path)
        
        # Valida existência do modelo
        if not model_file.exists():
            raise FileNotFoundError(
                f"Modelo de landmarks não encontrado: {model_path}"
            )
        
        # Determina backend automaticamente pela extensão se for pytorch
        if backend.lower() in ['pytorch', 'yolo']:
            backend_detected = LandmarksModelFactory._detect_backend(model_file)
        else:
            backend_detected = backend.lower()
        
        # FALLBACK: TensorRT e OpenVINO ainda não suportados para landmarks
        # Usa PyTorch automaticamente com aviso no log
        if backend_detected in ['tensorrt', 'openvino']:
            logger.warning(
                f"⚠️ Backend '{backend_detected}' não suportado para landmarks. "
                f"Usando PyTorch como fallback. "
                f"(TensorRT e OpenVINO para landmarks serão implementados em futuras versões)"
            )
            backend_detected = 'yolo'
        
        # Cria adapter apropriado
        if backend_detected == 'yolo':
            return LandmarksYOLOModelAdapter(model_path, device, image_size)
    
    @staticmethod
    def _detect_backend(model_file: Path) -> str:
        """
        Detecta o backend apropriado baseado na extensão do arquivo.
        
        :param model_file: Path do arquivo do modelo.
        :return: Nome do backend ('yolo', 'tensorrt', 'openvino').
        """
        extension = model_file.suffix.lower()
        
        if extension == '.pt':
            return 'yolo'
        elif extension == '.engine':
            return 'tensorrt'
        elif extension == '.xml':
            return 'openvino'
        else:
            # Padrão: tenta YOLO
            return 'yolo'
    
    @staticmethod
    def is_supported(model_path: str) -> bool:
        """
        Verifica se o modelo é suportado pela factory.
        
        :param model_path: Caminho para o modelo.
        :return: True se suportado, False caso contrário.
        """
        try:
            model_file = Path(model_path)
            if not model_file.exists():
                return False
            
            backend = LandmarksModelFactory._detect_backend(model_file)
            
            # Por enquanto, apenas YOLO é suportado
            return backend == 'yolo'
        
        except Exception:
            return False
