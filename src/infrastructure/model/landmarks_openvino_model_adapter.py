# src/infrastructure/model/landmarks_openvino_model_adapter.py
"""
Adapter para modelos OpenVINO de detecção de landmarks faciais.
Implementa a interface ILandmarksModel usando OpenVINO.
"""

from typing import Optional, Tuple, List
import numpy as np
import logging
from ultralytics import YOLO

from src.domain.services.landmarks_model_interface import ILandmarksModel

logger = logging.getLogger(__name__)


class LandmarksOpenVINOModelAdapter(ILandmarksModel):
    """
    Adapter que encapsula um modelo OpenVINO para detecção de landmarks faciais.
    Processa crops de faces e retorna landmarks (keypoints).
    """
    
    def __init__(self, model_path: str, device: str = 'cpu', image_size: int = 640):
        """
        Inicializa o adapter com um modelo OpenVINO.
        
        :param model_path: Caminho para o arquivo do modelo OpenVINO (.xml).
        :param device: Dispositivo padrão para inferência (cpu, gpu, etc).
        :param image_size: Tamanho de imagem para inferência (640 ou 1280).
        """
        self.model_path = model_path
        self.device = device
        self.image_size = image_size
        
        try:
            # Carrega modelo OpenVINO
            self.model = YOLO(model_path)
            logger.info(f"Modelo OpenVINO de landmarks carregado: {model_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo OpenVINO: {e}")
            raise
        
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
        Normaliza o tamanho de todas as imagens no batch para 640x480 pixels.
        Cria uma imagem com fundo preto e coloca o crop no canto superior esquerdo.
        
        :param face_crops: Lista de crops de face com possivelmente diferentes tamanhos.
        :return: Lista de crops normalizados para 640x480 pixels.
        """
        if not face_crops:
            return []
        
        normalized = []
        target_width = 640
        target_height = 480
        
        for img in face_crops:
            if img is None or img.size == 0:
                # Cria uma imagem preta 640x480 para crops inválidos
                black_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                normalized.append(black_image)
            else:
                # Cria canvas preto 640x480
                canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                
                # Obtém dimensões do crop
                crop_height, crop_width = img.shape[:2]
                
                # Copia o crop para o canto superior esquerdo da canvas
                # Limita ao tamanho máximo da canvas se necessário
                end_height = min(crop_height, target_height)
                end_width = min(crop_width, target_width)
                
                canvas[0:end_height, 0:end_width] = img[0:end_height, 0:end_width]
                normalized.append(canvas)
        
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
        :param device: Device para inferência (cpu, gpu, etc). Se None, usa device padrão.
        :return: Tupla (landmarks, confidence) ou None.
        """
        if face_crop.size == 0:
            return None
        
        try:
            # OpenVINO device já compilado no modelo, não alteramos aqui
            # Executa inferência com imgsz configurado
            results = self.model(face_crop, conf=conf, verbose=False, imgsz=self.image_size)
            
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
                logger.error(f"Erro na inferência de landmarks OpenVINO: {e}")
            return None
    
    def predict_batch(
        self,
        face_crops: List[np.ndarray],
        conf: float = 0.5,
        verbose: bool = False,
        device=None
    ) -> List[Optional[Tuple[np.ndarray, float]]]:
        """
        Detecta landmarks em múltiplos crops de face usando OpenVINO (batch).
        OTIMIZAÇÃO: Processa múltiplas faces em um único batch.
        
        :param face_crops: Lista de crops de face (numpy arrays BGR).
        :param conf: Threshold de confiança mínima.
        :param verbose: Se deve exibir logs detalhados.
        :param device: Device para inferência (não usado para OpenVINO pré-compilado).
        :return: Lista de tuplas (landmarks_array, confidence) ou None para cada crop.
        """
        if not face_crops:
            return []
        
        try:
            # CORREÇÃO: Normaliza tamanho de todas as imagens antes do batch inference
            # OpenVINO falha quando imagens têm dimensões diferentes no batch
            normalized_crops = self._normalize_batch_sizes(face_crops)
            
            # OpenVINO aceita lista de imagens para batch inference com imgsz configurado
            results = self.model(normalized_crops, conf=conf, verbose=False, imgsz=self.image_size)
            
            # Processa resultados
            landmarks_list = []
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    landmarks_list.append(None)
                    continue
                
                box = result.boxes[0]
                confidence = float(box.conf[0])
                
                if result.keypoints is None or len(result.keypoints) == 0:
                    landmarks_list.append(None)
                    continue
                
                landmarks = result.keypoints[0].xy.cpu().numpy()
                
                if landmarks.shape[0] == 0:
                    landmarks_list.append(None)
                else:
                    landmarks_list.append((landmarks, confidence))
            
            return landmarks_list
            
        except Exception as e:
            if verbose:
                logger.error(f"Erro na inferência em batch OpenVINO: {e}")
            return [None] * len(face_crops)

    def get_model_info(self) -> dict:
        """
        Retorna informações sobre o modelo OpenVINO de landmarks.
        
        :return: Dicionário com informações do modelo.
        """
        return {
            'model_path': self.model_path,
            'backend': 'OpenVINO',
            'device': self.device,
            'num_keypoints': self._num_keypoints,
            'precision': 'OpenVINO Optimized',
            'format': 'OpenVINO IR (.xml)',
            'optimization': 'Intel Hardware Optimized'
        }
    
    def get_num_keypoints(self) -> int:
        """
        Retorna o número de keypoints detectados pelo modelo.
        
        :return: Número de keypoints (ex: 5 para olhos, nariz, cantos da boca).
        """
        return self._num_keypoints
