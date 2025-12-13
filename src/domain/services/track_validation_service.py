"""
Serviço de domínio para validação de tracks (Domain Layer).
Contém regras de negócio para validar se um track é válido para envio.
"""

import logging
from typing import Tuple
from src.domain.entities import Track


class TrackValidationService:
    """
    Serviço de domínio responsável por validar tracks antes do envio ao FindFace.
    Aplica regras de negócio: movimento mínimo, confiança mínima, bbox mínimo.
    """

    def __init__(
        self,
        min_movement_threshold: float,
        min_movement_percentage: float,
        min_confidence_threshold: float,
        min_bbox_width: int
    ):
        """
        Inicializa o serviço de validação de tracks.
        
        :param min_movement_threshold: Distância mínima em pixels para considerar movimento.
        :param min_movement_percentage: Percentual mínimo de frames com movimento.
        :param min_confidence_threshold: Confiança mínima para enviar track.
        :param min_bbox_width: Largura mínima do bbox para enviar track.
        """
        self.min_movement_threshold = min_movement_threshold
        self.min_movement_percentage = min_movement_percentage
        self.min_confidence_threshold = min_confidence_threshold
        self.min_bbox_width = min_bbox_width
        self.logger = logging.getLogger(self.__class__.__name__)

    def is_valid(self, track: Track) -> Tuple[bool, str]:
        """
        Valida se um track atende aos critérios de qualidade.
        
        :param track: Track a ser validado.
        :return: Tupla (is_valid, reason).
                 is_valid: True se o track é válido, False caso contrário.
                 reason: Razão da invalidação (vazio se válido).
        """
        # 1. Verifica se track tem eventos
        if track.event_count == 0:
            return (False, "Track sem eventos")
        
        # 2. Verifica movimento significativo
        has_movement = track.has_movement
        movement_stats = track.get_movement_statistics()
        
        if not has_movement:
            return (
                False,
                f"Movimento insuficiente (distância média: {movement_stats['average_distance']:.2f}px, "
                f"% frames com movimento: {movement_stats['movement_percentage']:.1f}%)"
            )
        
        # 3. Verifica melhor evento
        best_event = track.get_best_event()
        if best_event is None:
            return (False, "Track sem melhor evento")
        
        # 4. Verifica confiança mínima
        best_confidence = best_event.confidence.value()
        if best_confidence < self.min_confidence_threshold:
            return (
                False,
                f"Confiança abaixo do mínimo ({best_confidence:.2f} < {self.min_confidence_threshold:.2f})"
            )
        
        # 5. Verifica largura mínima do bbox
        bbox = best_event.bbox.value()
        bbox_width = bbox[2] - bbox[0]
        if bbox_width < self.min_bbox_width:
            return (
                False,
                f"Bbox muito pequeno (largura: {bbox_width}px < {self.min_bbox_width}px)"
            )
        
        # Track válido!
        return (True, "")
