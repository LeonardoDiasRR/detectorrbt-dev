"""
Serviço de domínio para gerenciar o ciclo de vida de tracks (Domain Layer).
Contém regras de negócio para adicionar eventos, finalizar tracks, obter melhor evento.
"""

import logging
from typing import Optional
from src.domain.entities import Track, Event
from src.domain.value_objects import IdVO


class TrackLifecycleService:
    """
    Serviço de domínio responsável pelo ciclo de vida de tracks.
    Gerencia adição de eventos, finalização, seleção de melhor evento.
    """

    def __init__(self, max_frames_per_track: int):
        """
        Inicializa o serviço de ciclo de vida de tracks.
        
        :param max_frames_per_track: Máximo de frames/eventos por track.
        """
        self.max_frames_per_track = max_frames_per_track
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_event_to_track(self, track: Track, event: Event) -> bool:
        """
        Adiciona um evento ao track se ainda não atingiu o limite.
        
        :param track: Track ao qual adicionar o evento.
        :param event: Evento a ser adicionado.
        :return: True se adicionou, False se atingiu limite.
        """
        if track.event_count >= self.max_frames_per_track:
            self.logger.debug(
                f"Track {track.id.value()} atingiu limite de {self.max_frames_per_track} eventos"
            )
            return False
        
        track.add_event(event)
        return True

    def should_finalize_track(self, track: Track, frames_lost: int, max_frames_lost: int) -> bool:
        """
        Decide se um track deve ser finalizado.
        
        :param track: Track a verificar.
        :param frames_lost: Número de frames perdidos (track não detectado).
        :param max_frames_lost: Máximo de frames perdidos antes de finalizar.
        :return: True se deve finalizar, False caso contrário.
        """
        return frames_lost >= max_frames_lost

    def get_best_event(self, track: Track) -> Optional[Event]:
        """
        Obtém o melhor evento do track (maior qualidade facial).
        
        :param track: Track do qual obter o melhor evento.
        :return: Melhor evento ou None se track vazio.
        """
        return track.get_best_event()

    def get_track_summary(self, track: Track) -> dict:
        """
        Obtém resumo do track para logging.
        
        :param track: Track a sumarizar.
        :return: Dicionário com informações do track.
        """
        best_event = self.get_best_event(track)
        movement_stats = track.get_movement_statistics()
        
        return {
            'track_id': track.id.value(),
            'event_count': track.event_count,
            'has_movement': track.has_movement,
            'average_distance': movement_stats['average_distance'],
            'max_distance': movement_stats['max_distance'],
            'best_confidence': best_event.confidence.value() if best_event else 0.0,
            'best_quality': best_event.face_quality_score.value() if best_event else 0.0
        }
