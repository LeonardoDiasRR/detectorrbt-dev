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
    Detecta automaticamente a disponibilidade de TensorRT, OpenVINO e seleciona
    a melhor implementação disponível.
    
    Ordem de precedência (se GPU CUDA disponível):
    1. TensorRT (melhor performance em NVIDIA GPUs)
    2. OpenVINO (boa performance multi-plataforma)
    3. YOLO padrão (fallback)
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
        use_tensorrt: bool = True,
        tensorrt_precision: str = "FP16",
        tensorrt_workspace: int = 4,
        use_openvino: bool = True,
        openvino_device: str = "AUTO",
        openvino_precision: str = "FP16"
    ) -> IDetectionModel:
        """
        Cria uma instância de modelo de detecção.
        
        Ordem de tentativa:
        1. TensorRT (se habilitado, CUDA disponível e TensorRT instalado)
        2. OpenVINO (se habilitado e OpenVINO instalado)
        3. YOLO padrão (fallback)
        
        :param model_path: Caminho para o arquivo do modelo.
        :param use_tensorrt: Se deve tentar usar TensorRT (padrão: True).
        :param tensorrt_precision: Precisão do modelo TensorRT (FP16, FP32, INT8).
        :param tensorrt_workspace: Workspace em GB para TensorRT (padrão: 4GB).
        :param use_openvino: Se deve tentar usar OpenVINO (padrão: True).
        :param openvino_device: Dispositivo OpenVINO (AUTO, CPU, GPU, etc).
        :param openvino_precision: Precisão do modelo OpenVINO (FP16, FP32, INT8).
        :return: Instância de IDetectionModel.
        """
        from src.infrastructure.model.yolo_model_adapter import YOLOModelAdapter
        from src.infrastructure.model.openvino_model_adapter import OpenVINOModelAdapter
        from src.infrastructure.model.tensorrt_model_adapter import TensorRTModelAdapter
        
        model_path_obj = Path(model_path)
        
        # 1. Tenta TensorRT (melhor performance em GPUs NVIDIA)
        if use_tensorrt and ModelFactory.is_tensorrt_available():
            try:
                logger.info(
                    f"Tentando carregar modelo com TensorRT "
                    f"(precision={tensorrt_precision}, workspace={tensorrt_workspace}GB)"
                )
                return TensorRTModelAdapter(
                    model_path=str(model_path_obj),
                    precision=tensorrt_precision,
                    workspace=tensorrt_workspace
                )
            except Exception as e:
                logger.warning(
                    f"Falha ao carregar modelo com TensorRT: {e}. "
                    "Tentando fallback para OpenVINO ou YOLO padrão."
                )
        
        # 2. Tenta OpenVINO (boa performance multi-plataforma)
        if use_openvino and ModelFactory.is_openvino_available():
            try:
                logger.info(
                    f"Tentando carregar modelo com OpenVINO "
                    f"(device={openvino_device}, precision={openvino_precision})"
                )
                return OpenVINOModelAdapter(
                    model_path=str(model_path_obj),
                    device=openvino_device,
                    precision=openvino_precision
                )
            except Exception as e:
                logger.warning(
                    f"Falha ao carregar modelo com OpenVINO: {e}. "
                    "Fallback para implementação padrão YOLO."
                )
        
        # 3. Fallback: usa implementação padrão YOLO
        logger.info("Carregando modelo com implementação padrão YOLO")
        return YOLOModelAdapter(model_path=str(model_path_obj))

def create_yolo_model(model_path: str, **kwargs):
    with _model_init_lock:
        # dentro do lock fazemos a importação/instanciação que não é thread-safe
        model = YOLO(model_path, **kwargs)
    return model