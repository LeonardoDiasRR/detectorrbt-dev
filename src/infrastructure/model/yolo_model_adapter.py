# src/infrastructure/model/yolo_model_adapter.py
"""
Adaptador para modelos YOLO padrão (PyTorch).
"""

import logging
from typing import Iterator, Any

from ultralytics import YOLO

from src.domain.services.model_interface import IDetectionModel


logger = logging.getLogger(__name__)


class YOLOModelAdapter(IDetectionModel):
    """
    Adaptador para modelos YOLO usando PyTorch.
    Implementação padrão sem otimizações específicas.
    """
    
    def __init__(self, model_path: str):
        """
        Inicializa o adaptador YOLO.
        
        :param model_path: Caminho para o arquivo do modelo.
        """
        import torch
        
        self.model_path = model_path
        self._model = YOLO(model_path)
        
        # Configura FP16 se GPU CUDA estiver disponível
        self.use_fp16 = torch.cuda.is_available()
        if self.use_fp16:
            self._model.to('cuda')
            logger.info(f"Modelo YOLO carregado com FP16 em GPU CUDA: {model_path}")
        else:
            logger.info(f"Modelo YOLO carregado em CPU: {model_path}")
    
    def track(
        self,
        source: str,
        tracker: str,
        persist: bool = True,
        conf: float = 0.1,
        iou: float = 0.2,
        show: bool = False,
        stream: bool = True,
        batch: int = 4,
        verbose: bool = False,
        imgsz: int = 640
    ) -> Iterator[Any]:
        """
        Realiza tracking usando YOLO padrão.
        """
        return self._model.track(
            source=source,
            tracker=tracker,
            persist=persist,
            conf=conf,
            iou=iou,
            show=show,
            stream=stream,
            batch=batch,
            verbose=verbose,
            half=self.use_fp16,
            imgsz=imgsz
        )
    
    def get_model_info(self) -> dict:
        """
        Retorna informações sobre o modelo YOLO.
        """
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        precision = "FP16" if self.use_fp16 else "FP32"
        
        return {
            "type": "YOLO",
            "backend": "PyTorch",
            "model_path": self.model_path,
            "device": device,
            "precision": precision,
            "optimization": "FP16 (half precision)" if self.use_fp16 else "None"
        }