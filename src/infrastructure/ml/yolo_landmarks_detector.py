"""
Implementação concreta de ILandmarksDetector usando YOLO (Infrastructure Layer).
"""

from typing import Optional, Tuple, Any
import numpy as np
from src.domain.interfaces import ILandmarksDetector


class YOLOLandmarksDetector(ILandmarksDetector):
    """
    Implementação de ILandmarksDetector usando YOLO para detecção de landmarks faciais.
    Wrapper para o modelo YOLO de landmarks que segue a interface do domínio.
    """

    def __init__(self, model: Any):
        """
        Inicializa o detector de landmarks YOLO.
        
        :param model: Instância do modelo YOLO de landmarks.
        """
        self.model = model

    def predict(
        self,
        face_crop: np.ndarray,
        conf: float = 0.5,
        verbose: bool = False
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Detecta landmarks em um crop de face usando YOLO.
        
        :param face_crop: Crop da face (numpy array BGR).
        :param conf: Threshold de confiança mínima.
        :param verbose: Se deve exibir logs detalhados.
        :return: Tupla (landmarks_array, confidence) ou None se não detectou.
        """
        if face_crop.size == 0:
            return None
        
        # Chama predict do modelo YOLO de landmarks
        result = self.model.predict(
            face_crop=face_crop,
            conf=conf,
            verbose=verbose
        )
        
        return result  # Retorna (landmarks_array, confidence) ou None

    def get_model_info(self) -> dict:
        """
        Obtém informações sobre o modelo YOLO de landmarks.
        
        :return: Dicionário com informações do modelo.
        """
        return self.model.get_model_info()
