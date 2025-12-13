"""
Interface para detectores de landmarks faciais (Domain Layer).
Define o contrato que implementações concretas devem seguir.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class ILandmarksDetector(ABC):
    """
    Interface abstrata para detectores de landmarks faciais.
    Implementações concretas podem usar YOLO, MediaPipe, Dlib, etc.
    """

    @abstractmethod
    def predict(
        self,
        face_crop: np.ndarray,
        conf: float = 0.5,
        verbose: bool = False
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Detecta landmarks em um crop de face.
        
        :param face_crop: Crop da face (numpy array BGR).
        :param conf: Threshold de confiança mínima.
        :param verbose: Se deve exibir logs detalhados.
        :return: Tupla (landmarks_array, confidence) ou None se não detectou.
                 landmarks_array tem shape (num_keypoints, 2) com coordenadas (x, y).
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """
        Obtém informações sobre o modelo de landmarks.
        
        :return: Dicionário com informações do modelo (backend, device, num_keypoints, etc.).
        """
        pass
