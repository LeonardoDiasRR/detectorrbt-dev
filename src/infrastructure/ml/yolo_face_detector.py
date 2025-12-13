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

    def __init__(self, model: IDetectionModel):
        """
        Inicializa o detector YOLO.
        
        :param model: Instância do modelo YOLO (IDetectionModel).
        """
        if not isinstance(model, IDetectionModel):
            raise TypeError("model deve implementar IDetectionModel")
        
        self.model = model

    def detect_and_track(
        self,
        source: str,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        inference_size: Optional[Tuple[int, int]] = None,
        batch: int = 1,
        show: bool = False
    ) -> Generator[Any, None, None]:
        """
        Detecta e rastreia faces usando YOLO.track().
        
        :param source: URL do stream (RTSP, arquivo de vídeo, webcam, etc.).
        :param conf_threshold: Threshold de confiança para detecções.
        :param iou_threshold: Threshold de IoU para NMS.
        :param inference_size: Tamanho de inferência (width, height).
        :param batch: Tamanho do batch para processamento.
        :param show: Se True, exibe o vídeo em tempo real.
        :return: Generator que produz resultados de detecção YOLO.
        """
        # Prepara parâmetros para o modelo YOLO
        track_params = {
            'source': source,
            'tracker': 'bytetrack.yaml',
            'persist': True,  # Mantém IDs entre frames
            'conf': conf_threshold,
            'iou': iou_threshold,
            'verbose': False,
            'stream': True,  # Retorna generator
            'batch': batch,
            'show': show
        }
        
        # Adiciona inference_size se fornecido
        if inference_size is not None:
            track_params['imgsz'] = inference_size
        
        # Chama .track() do YOLO (retorna generator)
        for result in self.model.track(**track_params):
            yield result

    def get_model_info(self) -> dict:
        """
        Obtém informações sobre o modelo YOLO.
        
        :return: Dicionário com informações do modelo.
        """
        return self.model.get_model_info()
