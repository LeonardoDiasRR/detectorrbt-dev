"""
Serviço de domínio para detecção de movimento (Domain Layer).
Contém regras de negócio para detectar movimento significativo em tracks.
"""

import logging
import numpy as np
from typing import Dict, List


class MovementDetectionService:
    """
    Serviço de domínio responsável por detectar movimento significativo em tracks.
    Analisa mudanças de posição entre frames consecutivos.
    """

    def __init__(self, min_movement_threshold: float):
        """
        Inicializa o serviço de detecção de movimento.
        
        :param min_movement_threshold: Distância mínima em pixels para considerar movimento.
        """
        self.min_movement_threshold = min_movement_threshold
        self.logger = logging.getLogger(self.__class__.__name__)

    def calculate_movement_distance(
        self,
        bbox_previous: List[float],
        bbox_current: List[float]
    ) -> float:
        """
        Calcula a distância euclidiana entre os centros de dois bboxes.
        
        :param bbox_previous: Bbox anterior [x1, y1, x2, y2].
        :param bbox_current: Bbox atual [x1, y1, x2, y2].
        :return: Distância em pixels.
        """
        # Calcula centros dos bboxes
        center_prev = np.array([
            (bbox_previous[0] + bbox_previous[2]) / 2,
            (bbox_previous[1] + bbox_previous[3]) / 2
        ])
        
        center_curr = np.array([
            (bbox_current[0] + bbox_current[2]) / 2,
            (bbox_current[1] + bbox_current[3]) / 2
        ])
        
        # Distância euclidiana
        distance = np.linalg.norm(center_curr - center_prev)
        return float(distance)

    def has_significant_movement(
        self,
        bbox_previous: List[float],
        bbox_current: List[float]
    ) -> bool:
        """
        Verifica se houve movimento significativo entre dois frames.
        
        :param bbox_previous: Bbox anterior [x1, y1, x2, y2].
        :param bbox_current: Bbox atual [x1, y1, x2, y2].
        :return: True se houve movimento significativo, False caso contrário.
        """
        distance = self.calculate_movement_distance(bbox_previous, bbox_current)
        return distance >= self.min_movement_threshold

    def analyze_track_movement(
        self,
        bboxes: List[List[float]]
    ) -> Dict[str, float]:
        """
        Analisa o movimento ao longo de uma sequência de bboxes.
        
        :param bboxes: Lista de bboxes [[x1, y1, x2, y2], ...].
        :return: Dicionário com estatísticas de movimento.
        """
        if len(bboxes) < 2:
            return {
                'average_distance': 0.0,
                'max_distance': 0.0,
                'movement_count': 0,
                'total_frames': len(bboxes),
                'movement_percentage': 0.0
            }
        
        distances = []
        movement_count = 0
        
        for i in range(1, len(bboxes)):
            distance = self.calculate_movement_distance(bboxes[i-1], bboxes[i])
            distances.append(distance)
            
            if distance >= self.min_movement_threshold:
                movement_count += 1
        
        return {
            'average_distance': np.mean(distances) if distances else 0.0,
            'max_distance': np.max(distances) if distances else 0.0,
            'movement_count': movement_count,
            'total_frames': len(bboxes),
            'movement_percentage': (movement_count / (len(bboxes) - 1)) * 100 if len(bboxes) > 1 else 0.0
        }
