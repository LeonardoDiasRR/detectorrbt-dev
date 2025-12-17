"""
Implementação concreta de ILandmarksDetector usando YOLO (Infrastructure Layer).
"""

from typing import Optional, Tuple, Any, List
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
        verbose: bool = False,
        device=None
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Detecta landmarks em um crop de face usando YOLO.
        
        :param face_crop: Crop da face (numpy array BGR).
        :param conf: Threshold de confiança mínima.
        :param verbose: Se deve exibir logs detalhados.
        :param device: Device para inferência (int, str ou lista). Ex: 0, "0", [0, 1], "0,1". Se None, usa default.
        :return: Tupla (landmarks_array, confidence) ou None se não detectou.
        """
        if face_crop.size == 0:
            return None
        
        # Chama predict do modelo YOLO de landmarks
        result = self.model.predict(
            face_crop=face_crop,
            conf=conf,
            verbose=verbose,
            device=device
        )
        
        return result  # Retorna (landmarks_array, confidence) ou None
    
    def predict_batch(
        self,
        face_crops: List[np.ndarray],
        conf: float = 0.5,
        verbose: bool = False,
        device=None
    ) -> List[Optional[Tuple[np.ndarray, float]]]:
        """
        Detecta landmarks em múltiplos crops de face usando YOLO (batch).
        OTIMIZAÇÃO: Processa múltiplas faces em um único batch para GPU.
        
        :param face_crops: Lista de crops de face (numpy arrays BGR).
        :param conf: Threshold de confiança mínima.
        :param verbose: Se deve exibir logs detalhados.
        :param device: Device para inferência (int, str ou lista). Ex: 0, "0", [0, 1], "0,1". Se None, usa default.
        :return: Lista de tuplas (landmarks_array, confidence) ou None para cada crop.
        """
        if not face_crops:
            return []
        
        # Chama predict_batch do modelo YOLO de landmarks
        return self.model.predict_batch(
            face_crops=face_crops,
            conf=conf,
            verbose=verbose,
            device=device
        )

    def get_model_info(self) -> dict:
        """
        Obtém informações sobre o modelo YOLO de landmarks.
        
        :return: Dicionário com informações do modelo.
        """
        return self.model.get_model_info()
    
    def get_num_keypoints(self) -> int:
        """
        Retorna o número de keypoints (landmarks) que o modelo detecta.
        
        :return: Número de landmarks.
        """
        return self.model.get_num_keypoints()
