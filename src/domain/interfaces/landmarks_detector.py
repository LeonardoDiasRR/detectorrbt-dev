"""
Interface para detectores de landmarks faciais (Domain Layer).
Define o contrato que implementações concretas devem seguir.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
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
    def predict_batch(
        self,
        face_crops: List[np.ndarray],
        conf: float = 0.5,
        verbose: bool = False
    ) -> List[Optional[Tuple[np.ndarray, float]]]:
        """
        Detecta landmarks em múltiplos crops de face (batch).
        OTIMIZAÇÃO: Processa múltiplas faces em um único batch para GPU.
        
        :param face_crops: Lista de crops de face (numpy arrays BGR).
        :param conf: Threshold de confiança mínima.
        :param verbose: Se deve exibir logs detalhados.
        :return: Lista de tuplas (landmarks_array, confidence) ou None para cada crop.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """
        Obtém informações sobre o modelo de landmarks.
        
        :return: Dicionário com informações do modelo (backend, device, num_keypoints, etc.).
        """
        pass
