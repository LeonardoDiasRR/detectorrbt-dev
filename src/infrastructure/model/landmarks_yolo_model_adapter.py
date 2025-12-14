# src/infrastructure/model/landmarks_yolo_model_adapter.py
"""
Adapter para modelos YOLO de detecção de landmarks faciais.
Implementa a interface ILandmarksModel usando Ultralytics YOLO.
"""

from typing import Optional, Tuple, List
import numpy as np
from ultralytics import YOLO

from src.domain.services.landmarks_model_interface import ILandmarksModel


class LandmarksYOLOModelAdapter(ILandmarksModel):
    """
    Adapter que encapsula um modelo YOLO para detecção de landmarks faciais.
    Processa crops de faces e retorna landmarks (keypoints).
    """
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Inicializa o adapter com um modelo YOLO.
        
        :param model_path: Caminho para o arquivo do modelo YOLO (.pt).
        :param device: Dispositivo para inferência ('cpu', 'cuda', 'cuda:0', etc).
        """
        self.model_path = model_path
        self.device = device
        self.model = YOLO(model_path)
        
        # Move modelo para o dispositivo especificado
        self.model.to(device)
        
        # Detecta número de keypoints do modelo
        self._num_keypoints = self._detect_num_keypoints()
    
    def _detect_num_keypoints(self) -> int:
        """
        Detecta o número de keypoints que o modelo retorna.
        Faz uma inferência de teste em uma imagem dummy.
        
        :return: Número de keypoints.
        """
        # Cria imagem dummy pequena para teste
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
        
        try:
            results = self.model(dummy_image, verbose=False)
            if len(results) > 0 and results[0].keypoints is not None:
                if len(results[0].keypoints) > 0:
                    kpts = results[0].keypoints[0].xy.cpu().numpy()
                    if kpts.shape[0] > 0:
                        return kpts.shape[1]  # Número de keypoints
        except Exception:
            pass
        
        # Padrão: 5 keypoints (olhos, nariz, cantos da boca)
        return 5
    
    def predict(
        self,
        face_crop: np.ndarray,
        conf: float = 0.5,
        verbose: bool = False
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Executa inferência de landmarks no crop da face.
        
        :param face_crop: Imagem da face (crop) em formato BGR.
        :param conf: Threshold de confiança mínima.
        :param verbose: Se deve exibir logs detalhados.
        :return: Tupla (landmarks, confidence) ou None.
        """
        if face_crop.size == 0:
            return None
        
        try:
            # Executa inferência
            results = self.model(face_crop, conf=conf, verbose=verbose)
            
            # Valida resultados
            if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
                return None
            
            result = results[0]
            
            # Pega a primeira detecção (mais confiante)
            box = result.boxes[0]
            confidence = float(box.conf[0])
            
            # Extrai keypoints
            if result.keypoints is None or len(result.keypoints) == 0:
                return None
            
            landmarks = result.keypoints[0].xy.cpu().numpy()  # Shape: (N, 2)
            
            if landmarks.shape[0] == 0:
                return None
            
            return (landmarks, confidence)
            
        except Exception as e:
            if verbose:
                print(f"Erro na inferência de landmarks: {e}")
            return None
    
    def get_model_info(self) -> dict:
        """
        Retorna informações sobre o modelo de landmarks.
        
        :return: Dicionário com informações do modelo.
        """
        return {
            'model_path': self.model_path,
            'backend': 'YOLO',
            'device': self.device,
            'num_keypoints': self._num_keypoints,
            'precision': 'FP32',  # YOLO padrão usa FP32
            'format': 'PyTorch'
        }
    
    def get_num_keypoints(self) -> int:
        """
        Retorna o número de keypoints que o modelo detecta.
        
        :return: Número de landmarks.
        """
        return self._num_keypoints
    
    def predict_batch(
        self,
        face_crops: List[np.ndarray],
        conf: float = 0.5,
        verbose: bool = False
    ) -> List[Optional[Tuple[np.ndarray, float]]]:
        """
        Executa inferência em lote para múltiplos crops de face.
        OTIMIZAÇÃO: Processa múltiplas faces em um único batch para GPU.
        
        :param face_crops: Lista de crops de face (arrays numpy).
        :param conf: Confiança mínima para detecção.
        :param verbose: Se True, exibe informações de debug.
        :return: Lista de tuplas (landmarks, confidence) ou None para cada crop.
        """
        if not face_crops:
            return []
        
        try:
            # YOLO aceita lista de imagens para batch inference
            results = self.model(face_crops, conf=conf, verbose=verbose)
            
            batch_landmarks = []
            
            for result in results:
                # Valida resultado individual
                if result.boxes is None or len(result.boxes) == 0:
                    batch_landmarks.append(None)
                    continue
                
                # Pega a primeira detecção (mais confiante)
                box = result.boxes[0]
                confidence = float(box.conf[0])
                
                # Extrai keypoints
                if result.keypoints is None or len(result.keypoints) == 0:
                    batch_landmarks.append(None)
                    continue
                
                landmarks = result.keypoints[0].xy.cpu().numpy()  # Shape: (N, 2)
                
                if landmarks.shape[0] == 0:
                    batch_landmarks.append(None)
                    continue
                
                batch_landmarks.append((landmarks, confidence))
            
            return batch_landmarks
            
        except Exception as e:
            if verbose:
                print(f"Erro na inferência em lote de landmarks: {e}")
            # Retorna None para todos os crops em caso de erro
            return [None] * len(face_crops)
