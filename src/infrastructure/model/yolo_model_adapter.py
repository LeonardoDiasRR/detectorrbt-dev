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
        device=None,
        persist: bool = True,
        conf: float = 0.1,
        iou: float = 0.07,
        show: bool = False,
        stream: bool = True,
        batch: int = 4,
        verbose: bool = False,
        imgsz: int = 640
    ) -> Iterator[Any]:
        """
        Realiza tracking usando YOLO padrão.
        
        :param source: Fonte de vídeo (arquivo, stream RTSP, câmera, etc).
        :param tracker: Configuração do rastreador.
        :param device: Device para inferência (str: "0", "0,1", etc). Se None, usa padrão.
        :param persist: Se True, mantém IDs de tracks entre reinícios.
        :param conf: Threshold de confiança.
        :param iou: Threshold de IoU para NMS.
        :param show: Se deve exibir vídeo.
        :param stream: Se deve usar streaming.
        :param batch: Tamanho do batch.
        :param verbose: Se deve imprimir logs verbosos.
        :param imgsz: Tamanho de inferência.
        :return: Iterator com resultados de tracking.
        """
        # Move o modelo para o device especificado ANTES de chamar track()
        # Compatibilidade com versões antigas do YOLO que não suportam device em track()
        if device is not None:
            try:
                # Se device for lista, converte para primeiro elemento (string)
                if isinstance(device, (list, tuple)):
                    device = str(device[0])
                else:
                    device = str(device)
                
                # Se device contém vírgula (ex: "0,1"), pega primeira GPU
                if ',' in device:
                    device = device.split(',')[0].strip()
                
                # Converte para formato PyTorch válido
                # "0" → "cuda:0", "1" → "cuda:1", "cpu" → "cpu", "cuda:0" → "cuda:0"
                if device and device != "cpu":
                    if not device.startswith("cuda:"):
                        device = f"cuda:{device}"
                
                self._model.to(device)
            except Exception as e:
                logger.warning(f"Falha ao mover modelo para device {device}: {e}")
        
        # Se device não foi especificado, usa o padrão (None deixa YOLO decidir)
        track_kwargs = {
            "source": source,
            "tracker": tracker,
            "persist": persist,
            "conf": conf,
            "iou": iou,
            "show": show,
            "stream": stream,
            "batch": batch,
            "verbose": verbose,
            "half": self.use_fp16,
            "imgsz": imgsz
        }
        
        # NÃO adiciona device ao track_kwargs pois track() não aceita esse parâmetro
        # Device é gerenciado via model.to(device) acima
        
        track_kwargs["classes"] = [0]
        return self._model.track(**track_kwargs)
    
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