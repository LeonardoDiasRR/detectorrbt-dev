"""
Interface para detectores de faces (Domain Layer).
Define o contrato que implementações concretas devem seguir.
"""

from abc import ABC, abstractmethod
from typing import Generator, Any, Optional, Tuple


class IFaceDetector(ABC):
    """
    Interface abstrata para detectores de faces.
    Implementações concretas podem usar YOLO, MTCNN, RetinaFace, etc.
    """

    @abstractmethod
    def detect_and_track(
        self,
        source: str,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        tracker: str = "bytetrack.yaml",
        inference_size: Optional[Tuple[int, int]] = None,
        batch: int = 1,
        show: bool = False
    ) -> Generator[Any, None, None]:
        """
        Detecta e rastreia faces em um stream de vídeo.
        
        :param source: URL do stream (RTSP, arquivo de vídeo, webcam, etc.).
        :param conf_threshold: Threshold de confiança para detecções.
        :param iou_threshold: Threshold de IoU para NMS.
        :param tracker: Modelo de rastreamento (bytetrack.yaml ou deep_sort.yaml).
        :param inference_size: Tamanho de inferência (width, height).
        :param batch: Tamanho do batch para processamento.
        :param show: Se True, exibe o vídeo em tempo real.
        :return: Generator que produz resultados de detecção.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """
        Obtém informações sobre o modelo de detecção.
        
        :return: Dicionário com informações do modelo (backend, device, precision, etc.).
        """
        pass
