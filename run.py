# built-in
import os
import sys
import logging
import traceback
import threading
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from queue import Queue

# local
from src.infrastructure import ConfigLoader, AppSettings
from src.infrastructure.external.findface_client import create_findface_client
from src.infrastructure.repositories import CameraRepositoryFindface
from src.application.use_cases import CameraManager
from src.domain.adapters import FindfaceAdapter
from src.domain.services import ImageSaveService, FaceQualityService
from src.domain.services import (
    TrackValidationService,
    EventCreationService,
    MovementDetectionService,
    TrackLifecycleService
)
from src.infrastructure.model import ModelFactory
from src.infrastructure.model.landmarks_model_factory import LandmarksModelFactory
from src.infrastructure.ml import YOLOFaceDetector, YOLOLandmarksDetector

# Suprimir avisos do OpenCV e Ultralytics
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['YOLO_VERBOSE'] = 'False'
# Aplicar supressão mais segura de mensagens conhecidas enviadas para stderr
# - mantém stderr visível para erros críticos
# - filtra apenas linhas conhecidas, como 'Waiting for stream'
import warnings

# Filtrar avisos do módulo warnings que tenham mensagem específica
warnings.filterwarnings('ignore', message=r'.*Waiting for stream.*')

# Reduz nível de log para bibliotecas verbosas quando possível
logging.getLogger('ultralytics').setLevel(logging.ERROR)
logging.getLogger('ultralytics.engine').setLevel(logging.ERROR)


class _StderrFilter:
    """Wrapper para sys.stderr que filtra linhas específicas antes de escrevê-las.

    Mantém o comportamento original para a maior parte do output, apenas omite
    mensagens que contenham o texto 'Waiting for stream'. Isso evita suprimir
    outros erros importantes.
    """
    def __init__(self, orig):
        self._orig = orig

    def write(self, data):
        try:
            text = data.decode('utf-8', errors='ignore') if isinstance(data, (bytes, bytearray)) else str(data)
            # Filtra linhas indesejadas
            if 'Waiting for stream' in text:
                return
            # Passa adiante
            self._orig.write(data)
        except Exception:
            # Em caso de erro no filtro, escreve o conteúdo original para não perder mensagens
            try:
                self._orig.write(data)
            except Exception:
                pass

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass

    def fileno(self):
        return getattr(self._orig, 'fileno', lambda: None)()

    def isatty(self):
        return getattr(self._orig, 'isatty', lambda: False)()


# Substitui sys.stderr por um filtro leve apenas em plataformas Unix-like
try:
    if sys.platform.startswith('linux') or sys.platform == 'darwin':
        sys.stderr = _StderrFilter(sys.stderr)
except Exception:
    # Se ocorrer qualquer problema, não quebra a inicialização
    pass

# Configura logging com rotação
log_file = os.path.join(os.path.dirname(__file__), "detectorrbt.log")

# Limpa arquivo de log anterior (inicia com log limpo a cada execução)
if os.path.exists(log_file):
    try:
        os.remove(log_file)
    except Exception:
        pass  # Se não conseguir remover, continua (sobrescreve)

# Handler com rotação: 2MB por arquivo, máximo 3 backups (compactados)
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=2 * 1024 * 1024,  # 2MB
    backupCount=3,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)  # Valor padrão, será atualizado após carregar settings
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Handler para console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Valor padrão, será atualizado após carregar settings
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# OTIMIZAÇÃO: Sistema de logging assíncrono com fila para evitar I/O bloqueante
# Cria fila com capacidade de 10000 mensagens
log_queue = Queue(maxsize=10000)

# QueueListener processa logs em thread separada (não bloqueia thread principal)
queue_listener = QueueListener(
    log_queue,
    file_handler,
    console_handler,
    respect_handler_level=True
)
queue_listener.start()

# QueueHandler envia logs para a fila (operação instantânea, não bloqueante)
queue_handler = QueueHandler(log_queue)
queue_handler.setLevel(logging.INFO)  # Valor padrão, será atualizado após carregar settings

# Configura root logger para usar apenas o QueueHandler
# Remove formatters duplicados do root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)  # Valor padrão, será atualizado após carregar settings
root_logger.handlers.clear()  # Remove handlers existentes
root_logger.addHandler(queue_handler)


# run.py - topo (logo após imports básicos, ANTES de criar modelos/threads)
import importlib
import logging
import sys
import types

logger = logging.getLogger(__name__)

try:
    # força import sequencial dos módulos ultralytics que geram probleminhas
    import ultralytics
    # tenta importar os submódulos de callbacks de forma controlada
    importlib.import_module("ultralytics.utils.callbacks.base")
    importlib.import_module("ultralytics.utils.callbacks.platform")
    importlib.import_module("ultralytics.utils.callbacks.hub")
    logger.info("Ultralytics e callbacks importados com sucesso no processo principal.")
except Exception as e:
    logger.warning(f"Falha no pré-import completo do ultralytics: {e}. Aplicando fallback seguro.")

# Monkeypatch: torna add_integration_callbacks um no-op para evitar circular import em runtime
try:
    from ultralytics.utils import callbacks as _callbacks_pkg
    # garante existência do módulo base
    cb_base = getattr(_callbacks_pkg, "base", None) or getattr(sys.modules.get("ultralytics.utils.callbacks.base"), "__dict__", None)
    if cb_base:
        # define no-op de forma segura
        def _noop_add_integration_callbacks(self):
            return None
        # aplica patch somente se a função existir (ou atribui)
        setattr(_callbacks_pkg, "base", _callbacks_pkg.base if hasattr(_callbacks_pkg, "base") else types.ModuleType("ultralytics.utils.callbacks.base"))
        _callbacks_pkg.base.add_integration_callbacks = _noop_add_integration_callbacks
        logger.info("Callbacks do ultralytics substituídos por no-op para evitar import circular.")
except Exception as e:
    logger.warning(f"Não foi possível aplicar monkeypatch em callbacks do ultralytics: {e}")


def main(settings: AppSettings, findface_adapter: FindfaceAdapter):
    logger = logging.getLogger(__name__)
    
    # OTIMIZAÇÃO: Cria serviço assíncrono para salvamento de imagens
    # Compartilhado entre todas as câmeras para centralizar I/O
    image_save_service = ImageSaveService()
    logger.info("ImageSaveService iniciado (fila: 200)")
    
    # OTIMIZAÇÃO: Fila FindFace global compartilhada por todas as câmeras
    # Pool de N/2 workers (onde N = número de CPUs)
    import multiprocessing
    from queue import Queue
    
    num_cpus = multiprocessing.cpu_count()
    num_findface_workers = max(4, num_cpus // 2)  # Mínimo de 4 workers
    findface_queue = Queue(maxsize=500)  # Fila global
    findface_workers = []
    findface_workers_running = True
    
    logger.info(f"Criando fila FindFace global com {num_findface_workers} workers (CPUs: {num_cpus})")
    
    def findface_worker_func(worker_id, queue, adapter, running_flag_container):
        """Worker FindFace que processa eventos de todas as câmeras."""
        worker_logger = logging.getLogger(f"FindFaceWorker-{worker_id}")
        worker_logger.info(f"Worker {worker_id} iniciado")
        
        while running_flag_container[0]:
            try:
                event_data = queue.get(timeout=0.5)
                
                if event_data is None:  # Sinal de parada
                    worker_logger.info(f"Worker {worker_id} recebeu sinal de parada")
                    break
                
                camera_id, camera_name, track_id, event, total_events = event_data
                
                try:
                    resposta = adapter.send_event(event, track_id=track_id)
                    
                    if resposta:
                        worker_logger.info(
                            f"✓ FindFace - Melhor face do Track {track_id} enviada com sucesso! "
                            f"Camera: {camera_name} (ID: {camera_id}) | Total de eventos: {total_events} | "
                            f"Qualidade: {event.face_quality_score.value():.4f} | Confiança: {event.confidence.value():.4f}"
                        )
                    # else:
                    #     worker_logger.warning(
                    #         f"✗ FindFace - Resposta vazia - Camera {camera_name} (ID: {camera_id}) Track {track_id}"
                    #     )
                except Exception as e:
                    worker_logger.error(
                        f"✗ FindFace - FALHA - Camera {camera_name} (ID: {camera_id}) Track {track_id}: {e}"
                    )
                finally:
                    queue.task_done()
                    
            except Exception:
                if not running_flag_container[0]:
                    break
                continue
        
        worker_logger.info(f"Worker {worker_id} finalizado")
    
    # Container mutável para flag de running (compartilhado entre workers)
    findface_running_flag = [True]
    
    # Cria pool de workers FindFace
    for i in range(num_findface_workers):
        worker_thread = threading.Thread(
            target=findface_worker_func,
            args=(i + 1, findface_queue, findface_adapter, findface_running_flag),
            name=f"FindFaceWorker-{i + 1}",
            daemon=True
        )
        worker_thread.start()
        findface_workers.append(worker_thread)
    
    logger.info(f"Pool de {num_findface_workers} workers FindFace iniciado")
    
    # Limpa o diretório de imagens antes de iniciar
    imagens_dir = os.path.join(os.path.dirname(__file__), settings.storage.project_dir)
    if os.path.exists(imagens_dir):
        import shutil
        shutil.rmtree(imagens_dir)
        logger.info(f"Diretório '{imagens_dir}' limpo.")
    
    os.makedirs(imagens_dir, exist_ok=True)
    logger.info(f"Diretório '{imagens_dir}' criado.")
    
    # Obtém string de GPUs a usar ANTES de carregar modelos
    import torch
    gpu_devices = settings.processing.gpu_devices  # String: "0", "0,1", "cpu", etc
    
    # Verifica se há GPUs disponíveis no sistema
    cuda_available = torch.cuda.is_available()
    
    # Se CUDA não está disponível e gpu_devices não é "cpu", força uso de CPU
    if not cuda_available and gpu_devices != "cpu":
        logger.warning(f"⚠ CUDA não disponível - Forçando fallback para CPU (configurado: {gpu_devices})")
        gpu_devices = "cpu"
    elif cuda_available and gpu_devices != "cpu":
        logger.info(f"✓ CUDA disponível - Dispositivos: {gpu_devices}")
    elif gpu_devices == "cpu":
        logger.info(f"→ Usando CPU conforme configurado")
    else:
        logger.warning(f"⚠ CUDA não disponível - Modelos serão carregados em CPU")
    
    # Carrega detector de landmarks UMA VEZ para ser compartilhado entre todas as câmeras
    # Inferência SÍNCRONA em batch dentro de cada câmera
    # NOTA: Landmarks gerencia multi-GPU via device parameter em predict_batch()
    landmarks_detector = None
    try:
        logger.info(f"Iniciando carregamento do detector de landmarks...")
        logger.info(f"Modelo de landmarks: {settings.landmark.model_path}")
        
        from src.infrastructure.ml.yolo_landmarks_detector import YOLOLandmarksDetector
        from src.infrastructure.model.landmarks_model_factory import LandmarksModelFactory
        
        # Usa primeira GPU como device padrão (string format: "0")
        # Se for CPU, mantém "cpu"
        # Device completo (para multi-GPU) será passado em predict_batch()
        if gpu_devices == "cpu":
            landmarks_device = "cpu"
        else:
            landmarks_device = gpu_devices.split(',')[0] if gpu_devices else "0"
        
        # Cria o modelo de landmarks usando a factory
        landmarks_model = LandmarksModelFactory.create(
            model_path=settings.landmark.model_path,
            device=landmarks_device
        )
        
        # Cria o detector usando o modelo criado
        landmarks_detector = YOLOLandmarksDetector(model=landmarks_model)
        logger.info(f"✓ Detector de landmarks carregado: {settings.landmark.model_path}")
        logger.info(f"  Será usada distribuição de GPU via device parameter em predict_batch()")
    except Exception as e:
        logger.error(f"✗ Erro ao carregar detector de landmarks: {e}", exc_info=True)
        landmarks_detector = None
    
    # Limpa o diretório de imagens antes de iniciar
    imagens_dir = os.path.join(os.path.dirname(__file__), settings.storage.project_dir)
    if os.path.exists(imagens_dir):
        import shutil
        shutil.rmtree(imagens_dir)
        logger.info(f"Diretório '{imagens_dir}' limpo.")
    
    os.makedirs(imagens_dir, exist_ok=True)
    logger.info(f"Diretório '{imagens_dir}' criado.")
    
    # ============================================================================
    # SISTEMA DE GERENCIAMENTO DINÂMICO DE CÂMERAS (CameraManager)
    # ============================================================================
    logger.info("=" * 80)
    logger.info("Inicializando sistema de gerenciamento dinâmico de câmeras...")
    logger.info("=" * 80)
    
    # Cria repository de câmeras
    camera_repository = CameraRepositoryFindface(ff, camera_prefix=settings.findface.group_prefix)
    
    # Cria serviços de domínio compartilhados
    track_validation_service = TrackValidationService(
        min_movement_threshold=settings.track.min_movement_pixels,
        min_movement_percentage=settings.track.min_movement_percentage,
        min_confidence_threshold=settings.filter.min_confidence,
        min_bbox_area=settings.filter.min_bbox_area
    )
    
    event_creation_service = EventCreationService(
        peso_confianca=settings.face_quality.peso_confianca,
        peso_tamanho=settings.face_quality.peso_tamanho,
        peso_frontal=settings.face_quality.peso_frontal,
        peso_proporcao=settings.face_quality.peso_proporcao
    )
    
    movement_detection_service = MovementDetectionService(
        min_movement_threshold=settings.track.min_movement_pixels
    )
    
    track_lifecycle_service = TrackLifecycleService(
        max_frames_per_track=settings.bytetrack.max_frames,
        min_movement_threshold=settings.track.min_movement_pixels
    )
    
    face_quality_service = FaceQualityService() if settings.landmark.model_path else None
    
    # Define fábricas para criação de modelos e detectores
    def create_model():
        """Fábrica para criar modelos de detecção"""
        # NOVO: Não mais seleciona GPU específica - YOLO gerencia distribuição
        return ModelFactory.create_model(
            model_path=settings.yolo.model_path,
            use_tensorrt=settings.tensorrt.enabled,
            tensorrt_precision=settings.tensorrt.precision,
            tensorrt_workspace=settings.tensorrt.workspace,
            use_openvino=settings.openvino.enabled,
            openvino_device=settings.openvino.device,
            openvino_precision=settings.openvino.precision
        )
    
    def create_face_detector(model):
        """Fábrica para criar detectores de face a partir de um modelo"""
        return YOLOFaceDetector(
            model=model,
            persist=settings.yolo.persist
        )
    
    def create_landmarks_detector():
        """Fábrica para criar detectores de landmarks (compartilhados)"""
        return landmarks_detector
    
    # Cria CameraManager com monitoramento dinâmico
    camera_manager = CameraManager(
        camera_repository=camera_repository,
        findface_queue=findface_queue,
        findface_adapter=findface_adapter,
        settings=settings,
        event_creation_service=event_creation_service,
        movement_detection_service=movement_detection_service,
        track_validation_service=track_validation_service,
        track_lifecycle_service=track_lifecycle_service,
        face_quality_service=face_quality_service,
        model_factory=create_model,
        face_detector_factory=create_face_detector,
        landmarks_detector_factory=create_landmarks_detector,
        gpu_devices=gpu_devices
    )
    
    # Inicia monitoramento de câmeras (roda em background mesmo sem câmeras inicialmente)
    logger.info("Iniciando monitoramento de câmeras...")
    camera_manager.start_monitoring()
    
    logger.info("=" * 80)
    logger.info("✓ Sistema de gerenciamento dinâmico de câmeras iniciado!")
    logger.info(f"  - Intervalo de verificação: {settings.camera_monitoring.check_interval}s")
    logger.info(f"  - Janela de rastreamento de sucesso: {settings.camera_monitoring.success_tracking_window} eventos")
    logger.info(f"  - Taxa mínima de sucesso: {settings.camera_monitoring.min_success_rate * 100:.0f}%")
    logger.info("  - Câmeras serão adicionadas/removidas dinamicamente conforme disponibilidade no FindFace")
    logger.info("=" * 80)
    logger.info("Pressione Ctrl+C para parar o processamento.")
    
    # Mantém o programa principal rodando
    try:
        import time
        while True:
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupção detectada (Ctrl+C). Finalizando sistema...")
        
        # Para o CameraManager (finaliza todas as câmeras)
        logger.info("Finalizando gerenciador de câmeras...")
        camera_manager.stop_monitoring()
        logger.info("✓ Gerenciador de câmeras finalizado")
        
        # Finaliza workers FindFace globais
        logger.info("Finalizando pool de workers FindFace...")
        findface_running_flag[0] = False
        
        # Envia sinais de parada para todos os workers
        for i in range(num_findface_workers):
            try:
                findface_queue.put(None, timeout=0.5)
            except:
                pass
        
        # Aguarda workers FindFace finalizarem
        for i, worker in enumerate(findface_workers, 1):
            worker.join(timeout=2.0)
            if worker.is_alive():
                logger.warning(f"FindFace worker {i}/{num_findface_workers} não finalizou no tempo esperado")
            else:
                logger.info(f"FindFace worker {i}/{num_findface_workers} finalizado")
        
        logger.info("✓ Pool de workers FindFace finalizado")
        logger.info("✓ Aplicação encerrada com sucesso")


if __name__ == "__main__":
    try:
        # Carrega configurações type-safe
        settings = ConfigLoader.load()
        
        # Reconfigura logging com valores do settings
        log_level = getattr(logging, settings.logging.level, logging.INFO)
        log_format = logging.Formatter(settings.logging.format)
        
        file_handler.setLevel(log_level)
        file_handler.setFormatter(log_format)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(log_format)
        queue_handler.setLevel(log_level)
        root_logger.setLevel(log_level)
        
        logger.info(f"Nível de log configurado: {settings.logging.level}")
        
        # Cria cliente FindFace
        ff = create_findface_client(settings.findface)
        
        # Cria adapter do FindFace com qualidade JPEG configurável
        findface_adapter = FindfaceAdapter(
            ff, 
            camera_prefix=settings.findface.group_prefix,
            jpeg_quality=settings.compression.jpeg_quality
        )
        
        # Executa aplicação
        main(settings, findface_adapter)
        
    except Exception as e:
        print(f"Erro ao iniciar aplicação: {e}")
        print(traceback.format_exc())
    finally:
        # Para o queue listener antes de encerrar
        try:
            if 'queue_listener' in globals():
                queue_listener.stop()
                print("Sistema de logging assíncrono finalizado")
        except Exception as e:
            print(f"Erro ao finalizar queue listener: {e}")
        
        # Para o ImageSaveService
        try:
            if 'image_save_service' in locals():
                image_save_service.stop()
                print("ImageSaveService finalizado")
        except Exception as e:
            print(f"Erro ao finalizar ImageSaveService: {e}")
        
        if 'ff' in locals():
            ff.logout()