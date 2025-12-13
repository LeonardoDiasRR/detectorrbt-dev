"""
Caso de uso para carregar câmeras ativas do sistema.
Application Layer - orquestra a lógica de negócio.
"""

from typing import List
import logging
from src.domain.entities import Camera
from src.domain.repositories import CameraRepository
from src.infrastructure.config import AppSettings


class LoadCamerasUseCase:
    """
    Caso de uso para carregar câmeras ativas.
    Combina câmeras do repositório (FindFace) com câmeras extras do config.
    """

    def __init__(self, camera_repository: CameraRepository, settings: AppSettings):
        """
        Inicializa o caso de uso.

        :param camera_repository: Implementação do repositório de câmeras.
        :param settings: Configurações da aplicação.
        """
        self.camera_repository = camera_repository
        self.settings = settings
        self.logger = logging.getLogger(self.__class__.__name__)

    def execute(self) -> List[Camera]:
        """
        Executa o caso de uso para carregar câmeras ativas.
        
        :return: Lista de todas as câmeras ativas (FindFace + Config).
        """
        cameras = []

        # 1. Obtém câmeras ativas do FindFace
        try:
            findface_cameras = self.camera_repository.get_active_cameras()
            cameras.extend(findface_cameras)
            self.logger.info(f"Carregadas {len(findface_cameras)} câmeras do FindFace")
        except Exception as e:
            self.logger.error(f"Erro ao carregar câmeras do FindFace: {e}", exc_info=True)

        # 2. Adiciona câmeras extras do arquivo de configuração
        from src.domain.value_objects import IdVO, NameVO, CameraTokenVO, CameraSourceVO
        
        for cam_config in self.settings.cameras:
            camera = Camera(
                camera_id=IdVO(cam_config.id),
                camera_name=NameVO(cam_config.name),
                camera_token=CameraTokenVO(cam_config.token),
                source=CameraSourceVO(cam_config.url),
                active=True  # Câmeras do config são consideradas ativas
            )
            cameras.append(camera)
            self.logger.info(f"Câmera do config adicionada: {cam_config.name}")

        self.logger.info(f"Total de {len(cameras)} câmera(s) ativas carregadas")
        return cameras
