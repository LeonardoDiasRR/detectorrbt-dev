"""
Implementação concreta de IFaceDetector usando YOLO (Infrastructure Layer).
"""

from typing import Generator, Any, Optional, Tuple
from src.domain.interfaces import IFaceDetector
from src.domain.services.model_interface import IDetectionModel


class YOLOFaceDetector(IFaceDetector):
    """
    Implementação de IFaceDetector usando YOLO para detecção e tracking de faces.
    Wrapper para o modelo YOLO que segue a interface do domínio.
    """

    def __init__(self, model: IDetectionModel, persist: bool = False):
        """
        Inicializa o detector YOLO.
        
        :param model: Instância do modelo YOLO (IDetectionModel).
        :param persist: Se True, mantém IDs de tracks entre reinícios.
        """
        if not isinstance(model, IDetectionModel):
            raise TypeError("model deve implementar IDetectionModel")
        
        self.model = model
        self.persist = persist

    def detect_and_track(
        self,
        source: str,
        conf_threshold: float = 0.1,
        iou_threshold: float = 0.07,
        tracker: str = "bytetrack.yaml",
        device=None,
        inference_size: Optional[Tuple[int, int]] = None,
        batch: int = 1,
        show: bool = False
    ) -> Generator[Any, None, None]:
        """
        Detecta e rastreia faces usando YOLO.track().
        
        :param source: URL do stream (RTSP, arquivo de vídeo, webcam, etc.).
        :param conf_threshold: Threshold de confiança para detecções.
        :param iou_threshold: Threshold de IoU para NMS.
        :param tracker: Modelo de rastreamento (bytetrack.yaml ou deep_sort.yaml).
        :param device: Device(s) para inferência (int, str ou lista). Ex: 0, "0", [0, 1], "0,1"
                       Será aplicado via model.to(device) antes de chamar track().
        :param inference_size: Tamanho de inferência (width, height).
        :param batch: Tamanho do batch para processamento.
        :param show: Se True, exibe o vídeo em tempo real.
        :return: Generator que produz resultados de detecção YOLO.
        """
        # Determina tamanho de inferência
        imgsz = inference_size if inference_size is not None else 640
        
        # Constrói argumentos para track()
        track_kwargs = {
            "source": source,
            "tracker": tracker,
            "persist": self.persist,
            "conf": conf_threshold,
            "iou": iou_threshold,
            "verbose": False,
            "stream": True,
            "batch": batch,
            "show": show,
            "imgsz": imgsz
        }
        
        # Adiciona device se foi especificado
        # Será gerenciado pelo YOLOModelAdapter via model.to(device)
        if device is not None:
            track_kwargs["device"] = device
        
        # Chama .track() do YOLOModelAdapter 
        for result in self.model.track(**track_kwargs):
            yield result
