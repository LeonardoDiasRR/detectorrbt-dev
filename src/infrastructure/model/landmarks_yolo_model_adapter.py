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
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Inicializa o adapter com um modelo YOLO.
        
        :param model_path: Caminho para o arquivo do modelo YOLO (.pt).
        :param device: Dispositivo padrão para inferência ('cpu', 'cuda', 'cuda:0', etc). Usado apenas se não especificado em predict().
        """
        self.model_path = model_path
        self.device = device
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
    
    def _normalize_batch_sizes(self, face_crops: List[np.ndarray]) -> List[np.ndarray]:
        """
        Normaliza o tamanho de todas as imagens no batch para a mesma dimensão.
        Isso evita erros de tensor size mismatch no YOLO.
        
        :param face_crops: Lista de crops de face com possivelmente diferentes tamanhos.
        :return: Lista de crops redimensionados para o mesmo tamanho.
        """
        if not face_crops:
            return []
        
        # Se todas as imagens já têm o mesmo tamanho, retorna como está
        if len(face_crops) <= 1:
            return face_crops
        
        # Encontra a menor altura e largura comum
        heights = [img.shape[0] for img in face_crops if img is not None and img.size > 0]
        widths = [img.shape[1] for img in face_crops if img is not None and img.size > 0]
        
        if not heights or not widths:
            return face_crops
        
        # Usa tamanho de potência de 2 mais próximo para GPU optimization
        # YOLO funciona melhor com tamanhos múltiplos de 32
        min_height = min(heights)
        min_width = min(widths)
        
        # Alinha para múltiplo de 32 (stride padrão do YOLO)
        target_height = (min_height // 32) * 32
        target_width = (min_width // 32) * 32
        
        if target_height < 32:
            target_height = 32
        if target_width < 32:
            target_width = 32
        
        # Redimensiona todas as imagens (apenas se necessário)
        normalized = []
        for img in face_crops:
            if img is None or img.size == 0:
                normalized.append(img)
            elif img.shape[0] == target_height and img.shape[1] == target_width:
                # Já tem o tamanho correto
                normalized.append(img)
            else:
                # Redimensiona usando OpenCV (mais eficiente que numpy)
                try:
                    import cv2
                    resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                    normalized.append(resized)
                except ImportError:
                    # Se cv2 não disponível, usa resize do numpy
                    from scipy import ndimage
                    scale_y = target_height / img.shape[0]
                    scale_x = target_width / img.shape[1]
                    resized = ndimage.zoom(img, (scale_y, scale_x, 1), order=1)
                    normalized.append(resized.astype(np.uint8))
        
        return normalized

    
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
            # Determina device a usar (string format)
            inference_device = device if device is not None else self.device
            
            # Se device for lista, pega primeira GPU (PyTorch to() não aceita lista)
            if isinstance(inference_device, (list, tuple)):
                inference_device = str(inference_device[0])
            else:
                inference_device = str(inference_device)
            
            # Se device contém vírgula (ex: "0,1"), pega primeira GPU
            if ',' in inference_device:
                inference_device = inference_device.split(',')[0].strip()
            
            # Converte para formato PyTorch válido
            # "0" → "cuda:0", "1" → "cuda:1", "cpu" → "cpu", "cuda:0" → "cuda:0"
            if inference_device and inference_device != "cpu":
                if not inference_device.startswith("cuda:"):
                    inference_device = f"cuda:{inference_device}"
            
            # Move modelo para device especificado
            if inference_device is not None:
                try:
                    self.model.to(inference_device)
                except Exception as e:
                    logger.warning(f"Falha ao mover modelo para device {inference_device}: {e}")
            
            # Executa inferência (sem device parameter, já movido via model.to())
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
            # Determina device a usar (string format)
            inference_device = device if device is not None else self.device
            
            # Se device for lista, pega primeira GPU (PyTorch to() não aceita lista)
            if isinstance(inference_device, (list, tuple)):
                inference_device = str(inference_device[0])
            else:
                inference_device = str(inference_device)
            
            # Se device contém vírgula (ex: "0,1"), pega primeira GPU
            if ',' in inference_device:
                inference_device = inference_device.split(',')[0].strip()
            
            # Converte para formato PyTorch válido
            # "0" → "cuda:0", "1" → "cuda:1", "cpu" → "cpu", "cuda:0" → "cuda:0"
            if inference_device and inference_device != "cpu":
                if not inference_device.startswith("cuda:"):
                    inference_device = f"cuda:{inference_device}"
            
            # Move modelo para device especificado
            if inference_device is not None:
                try:
                    self.model.to(inference_device)
                except Exception as e:
                    logger.warning(f"Falha ao mover modelo para device {inference_device}: {e}")
            
            # CORREÇÃO: Normaliza tamanho de todas as imagens antes do batch inference
            # YOLO falha quando imagens têm dimensões diferentes no batch
            # Usa a menor dimensão para evitar padding excessivo
            normalized_crops = self._normalize_batch_sizes(face_crops)
            
            # YOLO aceita lista de imagens para batch inference (sem device parameter)
            results = self.model(normalized_crops, conf=conf, verbose=verbose)
            
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
