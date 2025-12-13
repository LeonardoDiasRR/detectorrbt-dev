# src/domain/services/landmarks_model_interface.py
"""
Interface abstrata para modelos de detecção de landmarks faciais.
Permite desacoplar a lógica de negócio da implementação específica do modelo.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class ILandmarksModel(ABC):
    """
    Interface para modelos de detecção de landmarks faciais.
    Encapsula a lógica de inferência de landmarks para permitir diferentes implementações.
    """
    
    @abstractmethod
    def predict(
        self,
        face_crop: np.ndarray,
        conf: float = 0.5,
        verbose: bool = False
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Executa inferência de landmarks em um crop de face.
        
        :param face_crop: Imagem da face (crop) em formato BGR (numpy array).
        :param conf: Threshold de confiança mínima para detecção.
        :param verbose: Se deve exibir logs detalhados.
        :return: Tupla (landmarks, confidence) ou None se nenhuma face for detectada.
                 - landmarks: array numpy de shape (N, 2) com coordenadas dos landmarks
                 - confidence: confiança da detecção (0.0 a 1.0)
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """
        Retorna informações sobre o modelo de landmarks carregado.
        
        :return: Dicionário com informações do modelo:
                 - 'model_path': caminho do modelo
                 - 'backend': tipo de backend (YOLO, TensorRT, OpenVINO, etc)
                 - 'device': dispositivo usado (cpu, cuda, etc)
                 - 'num_keypoints': número de landmarks retornados
                 - 'precision': precisão do modelo (FP16, FP32, etc)
        """
        pass
    
    @abstractmethod
    def get_num_keypoints(self) -> int:
        """
        Retorna o número de keypoints (landmarks) que o modelo detecta.
        
        :return: Número de landmarks (tipicamente 5 para modelos YOLO face).
        """
        pass
