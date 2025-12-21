# src/infrastructure/model/landmarks_yolo_model_adapter.py
"""
Adapter para modelos YOLO de detecção de landmarks faciais.
Implementa a interface ILandmarksModel usando Ultralytics YOLO.
"""

from typing import Optional, Tuple, List
import numpy as np
import logging
import torch
from ultralytics import YOLO

from src.domain.services.landmarks_model_interface import ILandmarksModel

logger = logging.getLogger(__name__)


class LandmarksYOLOModelAdapter(ILandmarksModel):
    """
    Adapter que encapsula um modelo YOLO para detecção de landmarks faciais.
    Processa crops de faces e retorna landmarks (keypoints).
    """
    
    def __init__(self, model_path: str, device: str = 'cpu', image_size: int = 640):
        """
        Inicializa o adapter com um modelo YOLO.
        
        :param model_path: Caminho para o arquivo do modelo YOLO (.pt).
        :param device: Dispositivo padrão para inferência ('cpu', 'cuda', 'cuda:0', etc). Usado apenas se não especificado em predict().
        :param image_size: Tamanho de imagem para inferência (640 ou 1280).
        """
        self.model_path = model_path
        self.device = device
        self.image_size = image_size
        self.model = YOLO(model_path)
        
        # Configura FP16 se GPU CUDA estiver disponível
        self.use_fp16 = torch.cuda.is_available()
        if self.use_fp16:
            self.model.to('cuda')
            logger.info(f"Modelo de landmarks carregado com FP16 em GPU CUDA")
        else:
            logger.info(f"Modelo de landmarks carregado com FP32 em CPU")
        
        # NÃO move modelo para device na inicialização
        # YOLO gerencia device dinamicamente via parâmetro em cada chamada predict()
        # Isso permite multi-GPU via device parameter passado per-call
        
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
        verbose: bool = False,
        device=None
    ) -> Optional[Tuple[np.ndarray, float]]:
        """
        Executa inferência de landmarks no crop da face.
        
        :param face_crop: Imagem da face (crop) em formato BGR.
        :param conf: Threshold de confiança mínima.
        :param verbose: Se deve exibir logs detalhados.
        :param device: Device para inferência (str: "0", "0,1", etc). Se None, usa device padrão.
        :return: Tupla (landmarks, confidence) ou None.
        """
        if face_crop.size == 0:
            return None
        
        try:
            # Determina device a usar
            inference_device = device if device is not None else self.device
            
            # Se device for lista, pega primeira GPU
            if isinstance(inference_device, (list, tuple)):
                inference_device = str(inference_device[0])
            else:
                inference_device = str(inference_device)
            
            # Se device contém vírgula, pega primeira GPU
            if ',' in inference_device:
                inference_device = inference_device.split(',')[0].strip()
            
            # Converte para formato YOLO válido
            if inference_device and inference_device != "cpu":
                if not inference_device.startswith("cuda"):
                    inference_device = int(inference_device)  # YOLO aceita int para GPU
            
            # Executa inferência passando device como parâmetro
            results = self.model(face_crop, conf=conf, verbose=verbose, imgsz=self.image_size, device=inference_device, classes=[0])
            
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
            'precision': 'FP16' if self.use_fp16 else 'FP32',
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
        verbose: bool = False,
        device=None
    ) -> List[Optional[Tuple[np.ndarray, float]]]:
        """
        Executa inferência em lote para múltiplos crops de face.
        OTIMIZAÇÃO: Processa múltiplas faces em um único batch para GPU.
        
        :param face_crops: Lista de crops de face (arrays numpy).
        :param conf: Confiança mínima para detecção.
        :param verbose: Se True, exibe informações de debug.
        :param device: Device para inferência (str: "0", "0,1", etc). Se None, usa device padrão.
        :return: Lista de tuplas (landmarks, confidence) ou None para cada crop.
        """
        if not face_crops:
            return []
        
        try:
            # Determina device a usar
            inference_device = device if device is not None else self.device
            
            # Se device for lista, pega primeira GPU
            if isinstance(inference_device, (list, tuple)):
                inference_device = str(inference_device[0])
            else:
                inference_device = str(inference_device)
            
            # Se device contém vírgula, pega primeira GPU
            if ',' in inference_device:
                inference_device = inference_device.split(',')[0].strip()
            
            # Converte para formato YOLO válido
            if inference_device and inference_device != "cpu":
                if not inference_device.startswith("cuda"):
                    inference_device = int(inference_device)  # YOLO aceita int para GPU
            
            # Processa crops passando device como parâmetro
            # em vez de chamar .to() que pode corromper o predictor
            results = self.model(
                face_crops, 
                conf=conf, 
                verbose=verbose, 
                imgsz=self.image_size,
                device=inference_device,
                classes=[0]
            )
            
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
                
                # IMPORTANTE: Remove dimensão extra do batch
                # Shape original: (1, N, 2) -> Shape final: (N, 2)
                landmarks = result.keypoints[0].xy.cpu().numpy()
                if landmarks.ndim == 3:  # Se tem 3 dimensões (batch, keypoints, coords)
                    landmarks = landmarks[0]  # Remove dimensão do batch
                
                if landmarks.shape[0] == 0:
                    batch_landmarks.append(None)
                    continue
                
                batch_landmarks.append((landmarks, confidence))
            
            return batch_landmarks
            
        except Exception as e:
            logger.error(f"Erro em predict_batch: {e}", exc_info=True)
            return [None] * len(face_crops)
