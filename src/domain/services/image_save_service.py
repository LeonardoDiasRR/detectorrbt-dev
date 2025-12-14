"""
Serviço de domínio para salvamento assíncrono de imagens.
Processa salvamentos em lote usando worker thread para evitar bloqueio da thread principal.
"""

import logging
import cv2
from queue import Queue, Empty
from threading import Thread
from pathlib import Path
from typing import Optional, Tuple
import numpy as np


class ImageSaveService:
    """
    Serviço de domínio para salvamento assíncrono de imagens.
    Usa fila e worker thread para processar salvamentos em background.
    """
    
    def __init__(self, queue_size: int = 200, camera_name: str = "Unknown"):
        """
        Inicializa o serviço de salvamento assíncrono.
        
        :param queue_size: Tamanho máximo da fila de salvamento.
        :param camera_name: Nome da câmera para identificação em logs.
        """
        self.queue_size = queue_size
        self.camera_name = camera_name
        self._save_queue: Queue = Queue(maxsize=queue_size)
        self._worker_running = True
        
        self.logger = logging.getLogger(f"ImageSaveService_{camera_name}")
        
        # Inicia worker automaticamente
        self._worker = Thread(
            target=self._save_worker,
            name=f"ImageSave-Worker-{self.camera_name}",
            daemon=True
        )
        self._worker.start()
        self.logger.info(
            f"Worker assíncrono de salvamento iniciado (fila: {self.queue_size})"
        )
    
    def stop(self):
        """Para o worker thread graciosamente."""
        if not self._worker_running:
            return
        
        self._worker_running = False
        
        # Envia sinal de parada
        try:
            self._save_queue.put(None, timeout=0.5)
        except:
            self.logger.warning("Não foi possível enviar sinal de parada para image save worker")
        
        # Aguarda worker finalizar
        if self._worker is not None:
            self._worker.join(timeout=3.0)
            
            if self._worker.is_alive():
                self.logger.warning("Image save worker não finalizou no tempo esperado")
            else:
                self.logger.info("Image save worker finalizado com sucesso")
    
    def save_async(
        self,
        image: np.ndarray,
        filepath: Path,
        jpeg_quality: int = 95
    ) -> bool:
        """
        Enfileira imagem para salvamento assíncrono.
        
        :param image: Array numpy da imagem a ser salva.
        :param filepath: Caminho completo para salvar a imagem.
        :param jpeg_quality: Qualidade JPEG (0-100).
        :return: True se enfileirado com sucesso, False se fila cheia.
        """
        if not self._worker_running:
            self.logger.error("Worker não está rodando. Chame start() primeiro.")
            return False
        
        try:
            # Enfileira para processamento assíncrono (non-blocking)
            self._save_queue.put_nowait((image, filepath, jpeg_quality))
            return True
            
        except:
            # Fila cheia - descarta imagem
            self.logger.warning(
                f"Fila de salvamento CHEIA ({self._save_queue.qsize()}/{self.queue_size}). "
                f"Imagem descartada: {filepath.name}"
            )
            return False
    
    def _save_worker(self):
        """Worker thread que processa fila de salvamento."""
        self.logger.info("Image save worker iniciado")
        
        saved_count = 0
        error_count = 0
        
        while self._worker_running:
            try:
                # Aguarda item da fila (timeout para verificar _worker_running)
                try:
                    item = self._save_queue.get(timeout=0.5)
                except Empty:
                    continue
                
                if item is None:  # Sinal de parada
                    self.logger.info("Image save worker recebeu sinal de parada")
                    break
                
                if not self._worker_running:
                    break
                
                image, filepath, jpeg_quality = item
                
                try:
                    # Cria diretório se não existir
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Salva imagem (operação bloqueante isolada na thread worker)
                    cv2.imwrite(
                        str(filepath),
                        image,
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                    )
                    
                    saved_count += 1
                    
                    # Log periódico a cada 50 imagens salvas
                    if saved_count % 50 == 0:
                        self.logger.info(
                            f"Progresso: {saved_count} imagens salvas, "
                            f"{error_count} erros, "
                            f"fila: {self._save_queue.qsize()}"
                        )
                    
                except Exception as e:
                    error_count += 1
                    self.logger.error(
                        f"✗ Erro ao salvar imagem {filepath.name}: {e}"
                    )
                
                finally:
                    self._save_queue.task_done()
                    
            except Exception as e:
                if not self._worker_running:
                    break
                self.logger.error(f"Erro no image save worker: {e}")
                continue
        
        self.logger.info(
            f"Image save worker finalizado. "
            f"Total: {saved_count} salvas, {error_count} erros"
        )
    
    def get_queue_size(self) -> int:
        """Retorna o tamanho atual da fila."""
        return self._save_queue.qsize()
    
    def is_running(self) -> bool:
        """Verifica se o worker está rodando."""
        return self._worker_running
