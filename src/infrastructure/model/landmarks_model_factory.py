# src/infrastructure/model/landmarks_model_factory.py
"""
Factory para criação de modelos de detecção de landmarks faciais.
Suporta diferentes backends (YOLO, TensorRT, OpenVINO).
"""

from pathlib import Path
from typing import Optional

from src.domain.services.landmarks_model_interface import ILandmarksModel
from src.infrastructure.model.landmarks_yolo_model_adapter import LandmarksYOLOModelAdapter


class LandmarksModelFactory:
    """
    Factory que cria instâncias de modelos de landmarks faciais
    baseado no tipo de arquivo e configuração.
    """
    
    @staticmethod
    def create(
        model_path: str,
        device: str = 'cpu',
        backend: Optional[str] = None
    ) -> ILandmarksModel:
        """
        Cria uma instância de ILandmarksModel baseado no modelo especificado.
        
        :param model_path: Caminho para o arquivo do modelo.
        :param device: Dispositivo para inferência ('cpu', 'cuda', etc).
        :param backend: Backend forçado ('yolo', 'tensorrt', 'openvino').
                       Se None, detecta automaticamente pela extensão.
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
        
        # Determina backend
        if backend is None:
            backend = LandmarksModelFactory._detect_backend(model_file)
        
        backend = backend.lower()
        
        # Cria adapter apropriado
        if backend == 'yolo':
            return LandmarksYOLOModelAdapter(model_path, device)
        
        elif backend == 'tensorrt':
            # TODO: Implementar TensorRT adapter para landmarks
            raise NotImplementedError(
                "TensorRT adapter para landmarks ainda não implementado. "
                "Use modelo YOLO (.pt) por enquanto."
            )
        
        elif backend == 'openvino':
            # TODO: Implementar OpenVINO adapter para landmarks
            raise NotImplementedError(
                "OpenVINO adapter para landmarks ainda não implementado. "
                "Use modelo YOLO (.pt) por enquanto."
            )
        
        else:
            raise ValueError(
                f"Backend '{backend}' não suportado para landmarks. "
                f"Suportados: yolo, tensorrt (futuro), openvino (futuro)"
            )
    
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
