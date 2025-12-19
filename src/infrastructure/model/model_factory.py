# src/infrastructure/model/model_factory.py
"""
Factory para criação de modelos de detecção.
Decide qual implementação usar baseado na disponibilidade do OpenVINO.
"""

import logging
from typing import Optional
from pathlib import Path
import threading

from src.domain.services.model_interface import IDetectionModel
from ultralytics import YOLO

logger = logging.getLogger(__name__)

_model_init_lock = threading.Lock()


class ModelFactory:
    """
    Factory responsável por criar instâncias de modelos de detecção.
    Seleciona a implementação do backend baseado na configuração.
    
    Backends suportados:
    1. pytorch - Implementação padrão com YOLO + PyTorch
    2. tensorrt - Otimização para GPUs NVIDIA com TensorRT
    3. openvino - Otimização multi-plataforma com OpenVINO
    """
    
    @staticmethod
    def is_cuda_available() -> bool:
        """
        Verifica se CUDA está disponível no sistema.
        
        :return: True se CUDA está disponível, False caso contrário.
        """
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"CUDA detectado: {torch.cuda.get_device_name(0)}")
                return True
            else:
                logger.info("CUDA não está disponível.")
                return False
        except ImportError:
            logger.info("PyTorch não está disponível. CUDA não pode ser detectado.")
            return False
    
    @staticmethod
    def is_tensorrt_available() -> bool:
        """
        Verifica se TensorRT está disponível no sistema.
        TensorRT requer CUDA para funcionar.
        
        :return: True se TensorRT está disponível, False caso contrário.
        """
        if not ModelFactory.is_cuda_available():
            return False
            
        try:
            import tensorrt
            logger.info(f"TensorRT detectado: versão {tensorrt.__version__}")
            return True
        except ImportError:
            logger.info("TensorRT não está disponível.")
            return False
    
    @staticmethod
    def is_openvino_available() -> bool:
        """
        Verifica se o OpenVINO está disponível no sistema.
        
        :return: True se OpenVINO está disponível, False caso contrário.
        """
        try:
            import openvino
            logger.info(f"OpenVINO detectado: versão {openvino.__version__}")
            return True
        except ImportError:
            logger.info("OpenVINO não está disponível.")
            return False
    
    @staticmethod
    def create_model(
        model_path: str,
        backend: str = "pytorch",
        precision: str = "FP16"
    ) -> IDetectionModel:
        """
        Cria uma instância de modelo de detecção.
        
        Ordem de tentativa baseada no backend configurado:
        1. tensorrt - Se backend=tensorrt, TensorRT está disponível e CUDA ativo
        2. openvino - Se backend=openvino e OpenVINO está instalado
        3. pytorch - Implementação padrão (fallback para qualquer backend)
        
        :param model_path: Caminho para o arquivo do modelo.
        :param backend: Backend a usar (pytorch, tensorrt ou openvino).
        :param precision: Precisão do modelo (FP16 ou FP32).
        :return: Instância de IDetectionModel.
        """
        from src.infrastructure.model.yolo_model_adapter import YOLOModelAdapter
        from src.infrastructure.model.openvino_model_adapter import OpenVINOModelAdapter
        from src.infrastructure.model.tensorrt_model_adapter import TensorRTModelAdapter
        
        model_path_obj = Path(model_path)
        
        # 1. Tenta TensorRT (se solicitado e disponível)
        if backend.lower() == "tensorrt":
            if ModelFactory.is_tensorrt_available():
                try:
                    logger.info(
                        f"Carregando modelo com TensorRT (precision={precision})"
                    )
                    return TensorRTModelAdapter(
                        model_path=str(model_path_obj),
                        precision=precision,
                        workspace=4
                    )
                except Exception as e:
                    logger.warning(
                        f"Falha ao carregar modelo com TensorRT: {e}. "
                        "Fallback para implementação padrão YOLO."
                    )
            else:
                logger.warning(
                    "TensorRT foi solicitado mas não está disponível. "
                    "Fallback para implementação padrão YOLO."
                )
        
        # 2. Tenta OpenVINO (se solicitado e disponível)
        elif backend.lower() == "openvino":
            if ModelFactory.is_openvino_available():
                try:
                    logger.info(
                        f"Carregando modelo com OpenVINO (device=AUTO, precision={precision})"
                    )
                    return OpenVINOModelAdapter(
                        model_path=str(model_path_obj),
                        device="AUTO",
                        precision=precision
                    )
                except Exception as e:
                    logger.warning(
                        f"Falha ao carregar modelo com OpenVINO: {e}. "
                        "Fallback para implementação padrão YOLO."
                    )
            else:
                logger.warning(
                    "OpenVINO foi solicitado mas não está disponível. "
                    "Fallback para implementação padrão YOLO."
                )
        
        # 3. Fallback: usa implementação padrão YOLO (para "pytorch" ou fallbacks)
        logger.info("Carregando modelo com implementação padrão YOLO")
        return YOLOModelAdapter(model_path=str(model_path_obj))

def create_yolo_model(model_path: str, **kwargs):
    with _model_init_lock:
        # dentro do lock fazemos a importação/instanciação que não é thread-safe
        model = YOLO(model_path, **kwargs)
    return model