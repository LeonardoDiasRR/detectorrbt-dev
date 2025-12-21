# src/infrastructure/model/tensorrt_model_adapter.py
"""
Adaptador para modelos YOLO otimizados com TensorRT.
TensorRT é a solução de inferência de alta performance da NVIDIA para GPUs CUDA.
"""

import logging
from typing import Iterator, Any
from pathlib import Path

from ultralytics import YOLO

from src.domain.services.model_interface import IDetectionModel


logger = logging.getLogger(__name__)


class TensorRTModelAdapter(IDetectionModel):
    """
    Adaptador para modelos YOLO usando TensorRT.
    Oferece a melhor performance em GPUs NVIDIA com CUDA.
    """
    
    def __init__(
        self,
        model_path: str,
        precision: str = "FP16",
        workspace: int = 4
    ):
        """
        Inicializa o adaptador TensorRT.
        
        :param model_path: Caminho para o arquivo do modelo.
        :param precision: Precisão do modelo (FP16, FP32, INT8).
        :param workspace: Workspace em GB para otimizações TensorRT (padrão: 4GB).
        """
        self.model_path = model_path
        self.precision = precision
        self.workspace = workspace
        self._model = self._load_tensorrt_model()
        
        logger.info(
            f"Modelo TensorRT carregado: {model_path} "
            f"(precision={precision}, workspace={workspace}GB)"
        )
    
    def _load_tensorrt_model(self) -> YOLO:
        """
        Carrega e exporta o modelo para TensorRT.
        
        O modelo TensorRT é exportado uma vez e reutilizado. Oferece a melhor
        performance em GPUs NVIDIA com CUDA.
        
        :return: Modelo YOLO otimizado com TensorRT.
        """
        model_path_obj = Path(self.model_path)
        
        # Determina o caminho do modelo TensorRT exportado
        tensorrt_model_path = model_path_obj.parent / f"{model_path_obj.stem}.engine"
        
        # Se o modelo TensorRT já existe, carrega diretamente
        if tensorrt_model_path.exists():
            logger.info(f"Modelo TensorRT já exportado encontrado: {tensorrt_model_path}")
            model = YOLO(str(tensorrt_model_path), task="detect")
            logger.info(f"Modelo TensorRT carregado com {len(model.names)} classes: {list(model.names.values())}")
        else:
            # Carrega o modelo original e exporta para TensorRT
            logger.info(f"Carregando modelo original: {self.model_path}")
            original_model = YOLO(self.model_path)
            
            logger.info(
                f"Exportando modelo para TensorRT "
                f"(precisão={self.precision}, workspace={self.workspace}GB)..."
            )
            
            # export() retorna o caminho do arquivo .engine
            export_path = original_model.export(
                format="engine",
                half=(self.precision == "FP16"),  # FP16 precision
                int8=(self.precision == "INT8"),  # INT8 precision
                workspace=self.workspace,  # Workspace em GB
                dynamic=False,  # Static shapes para melhor performance
                simplify=True,
                verbose=True
            )
            
            logger.info(f"Modelo exportado com sucesso para: {export_path}")
            
            # Recarrega o modelo exportado
            model = YOLO(export_path, task="detect")
            logger.info(f"Modelo TensorRT carregado com {len(model.names)} classes: {list(model.names.values())}")
        
        return model
    
    def track(
        self,
        source: str,
        tracker: str,
        persist: bool = True,
        conf: float = 0.1,
        iou: float = 0.07,
        show: bool = False,
        stream: bool = True,
        batch: int = 32,
        verbose: bool = False,
        imgsz: int = 640
    ) -> Iterator[Any]:
        """
        Realiza tracking usando modelo TensorRT otimizado.
        
        TensorRT oferece a melhor performance em GPUs NVIDIA.
        Batch padrão de 32 é recomendado para GPUs.
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
            imgsz=imgsz,
            classes=[0]
        ) # type: ignore
    
    def get_model_info(self) -> dict:
        """
        Retorna informações sobre o modelo TensorRT.
        """
        return {
            "type": "YOLO",
            "backend": "TensorRT",
            "model_path": self.model_path,
            "device": "cuda",
            "precision": self.precision,
            "optimization": f"TensorRT (workspace={self.workspace}GB)",
            "performance": "Maximum (NVIDIA GPU optimized)"
        }
