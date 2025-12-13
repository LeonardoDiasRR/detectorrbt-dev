# src/domain/services/model_interface.py
"""
Interface abstrata para modelos de detecção.
Permite desacoplar o ByteTrackDetectorService da implementação específica do modelo.
"""

from abc import ABC, abstractmethod
from typing import Any, Iterator


class IDetectionModel(ABC):
    """
    Interface para modelos de detecção de faces.
    Encapsula a lógica de tracking para permitir diferentes implementações.
    """
    
    @abstractmethod
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
        verbose: bool = False
    ) -> Iterator[Any]:
        """
        Realiza tracking em um stream de vídeo.
        
        :param source: Fonte do vídeo (URL RTSP, arquivo, etc).
        :param tracker: Configuração do tracker.
        :param persist: Se deve persistir os IDs entre frames.
        :param conf: Threshold de confiança.
        :param iou: Threshold de IOU.
        :param show: Se deve exibir o vídeo.
        :param stream: Se deve processar em modo stream.
        :param batch: Tamanho do batch.
        :param verbose: Se deve exibir logs detalhados.
        :return: Iterator de resultados de detecção.
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """
        Retorna informações sobre o modelo carregado.
        
        :return: Dicionário com informações do modelo (tipo, formato, dispositivo, etc).
        """
        pass