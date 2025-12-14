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
from src.application.use_cases import LoadCamerasUseCase, ProcessFaceDetectionUseCase
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

# Handler com rotação: 2MB por arquivo, máximo 3 backups (compactados)
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=2 * 1024 * 1024,  # 2MB
    backupCount=3,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Handler para console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
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
queue_handler.setLevel(logging.INFO)

# Configura root logger para usar apenas o QueueHandler
# Remove formatters duplicados do root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
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
                    resposta = adapter.send_event(event)
                    
                    if resposta:
                        worker_logger.info(
                            f"✓ FindFace - Melhor face do Track {track_id} enviada com sucesso! "
                            f"Camera: {camera_name} (ID: {camera_id}) | Total de eventos: {total_events}"
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
    
    processors = []
    
    # Obtém câmeras ativas usando DDD (Repository + Use Case)
    camera_repository = CameraRepositoryFindface(ff, camera_prefix=settings.findface.camera_prefix)
    load_cameras_use_case = LoadCamerasUseCase(camera_repository, settings)
    cameras_ff = load_cameras_use_case.execute()
    
    logger.info(f"Total de {len(cameras_ff)} câmera(s) ativas para processar.")
    
    # Obtém lista de GPUs a usar ANTES de carregar modelos
    import torch
    gpu_devices = settings.processing.gpu_devices
    num_gpus = len(gpu_devices)
    
    # Verifica se há GPUs disponíveis no sistema
    cuda_available = torch.cuda.is_available()
    if cuda_available and gpu_devices:
        landmarks_device = gpu_devices[0]
        logger.info(f"Distribuindo {len(cameras_ff)} câmera(s) entre {num_gpus} GPU(s): {gpu_devices}")
        logger.info(f"✓ CUDA disponível - Modelo de landmarks será carregado na GPU: {landmarks_device}")
    else:
        landmarks_device = "cpu"
        if not cuda_available:
            logger.warning(f"⚠ CUDA não disponível - Modelos serão carregados em CPU")
        logger.info(f"Distribuindo {len(cameras_ff)} câmera(s) em CPU")
        logger.info(f"Modelo de landmarks será carregado em CPU")
    
    # Carrega detector de landmarks UMA VEZ para ser compartilhado entre todas as câmeras
    # Inferência SÍNCRONA em batch dentro de cada câmera
    landmarks_detector = None
    try:
        logger.info(f"Iniciando carregamento do detector de landmarks...")
        logger.info(f"Modelo de landmarks: {settings.yolo.landmarks_model_path}")
        logger.info(f"Device para landmarks: {landmarks_device}")
        
        from src.infrastructure.ml.yolo_landmarks_detector import YOLOLandmarksDetector
        from src.infrastructure.model.landmarks_model_factory import LandmarksModelFactory
        
        logger.info(f"Imports realizados com sucesso")
        
        # Cria o modelo de landmarks usando a factory
        logger.info(f"Criando modelo de landmarks via factory...")
        landmarks_model = LandmarksModelFactory.create(
            model_path=settings.yolo.landmarks_model_path,
            device=landmarks_device
        )
        logger.info(f"Modelo de landmarks criado com sucesso")
        
        # Cria o detector usando o modelo criado
        logger.info(f"Criando detector de landmarks...")
        landmarks_detector = YOLOLandmarksDetector(model=landmarks_model)
        logger.info(f"✓ Detector de landmarks carregado (SÍNCRONO): {settings.yolo.landmarks_model_path}")
        logger.info(f"✓ Device usado: {landmarks_device}")
        logger.info(f"✓ landmarks_detector type: {type(landmarks_detector)}")
        logger.info(f"✓ landmarks_detector is None: {landmarks_detector is None}")
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
    
    processors = []
    
    # Obtém câmeras ativas usando DDD (Repository + Use Case)
    camera_repository = CameraRepositoryFindface(ff, camera_prefix=settings.findface.camera_prefix)
    load_cameras_use_case = LoadCamerasUseCase(camera_repository, settings)
    cameras_ff = load_cameras_use_case.execute()
    
    logger.info(f"Total de {len(cameras_ff)} câmera(s) ativas para processar.")
    
    # Cria serviços de domínio compartilhados
    track_validation_service = TrackValidationService(
        min_movement_threshold=settings.movement.min_movement_threshold_pixels,
        min_movement_percentage=settings.movement.min_movement_frame_percentage,
        min_confidence_threshold=settings.detection_filter.min_confidence,
        min_bbox_width=settings.detection_filter.min_bbox_width
    )
    
    # EventCreationService com pesos configurados (simplificado)
    event_creation_service = EventCreationService(
        peso_tamanho=settings.face_quality.peso_tamanho,
        peso_frontal=settings.face_quality.peso_frontal
    )
    
    movement_detection_service = MovementDetectionService(
        min_movement_threshold=settings.movement.min_movement_threshold_pixels
    )
    track_lifecycle_service = TrackLifecycleService(
        max_frames_per_track=settings.bytetrack.max_frames_per_track
    )
    
    face_quality_service = FaceQualityService() if settings.yolo.landmarks_model_path else None
    
    # Cria serviços de detecção - CADA CÂMERA COM SEU PRÓPRIO MODELO
    for i, camera in enumerate(cameras_ff, 1):
        try:
            # Determina device para esta câmera
            if cuda_available and gpu_devices:
                # Round-robin entre GPUs disponíveis
                gpu_id = gpu_devices[(i - 1) % num_gpus]
                logger.info(f"[{i}/{len(cameras_ff)}] Câmera {camera.camera_name.value()} → GPU {gpu_id}")
            else:
                gpu_id = "cpu"
                logger.info(f"[{i}/{len(cameras_ff)}] Câmera {camera.camera_name.value()} → CPU")
            
            # IMPORTANTE: Cria uma instância SEPARADA do modelo para cada câmera
            # Isso evita conflitos de thread-safety
            logger.info(f"[{i}/{len(cameras_ff)}] Carregando modelo para câmera {camera.camera_name.value()}...")
            
            # Configura GPU antes de criar o modelo (se disponível)
            if cuda_available and isinstance(gpu_id, int):
                torch.cuda.set_device(gpu_id)
                logger.info(f"[{i}/{len(cameras_ff)}] GPU {gpu_id} selecionada para câmera {camera.camera_name.value()}")
            
            detection_model = ModelFactory.create_model(
                model_path=settings.yolo.model_path,
                use_tensorrt=settings.tensorrt.enabled,
                tensorrt_precision=settings.tensorrt.precision,
                tensorrt_workspace=settings.tensorrt.workspace,
                use_openvino=settings.openvino.enabled,
                openvino_device=settings.openvino.device,
                openvino_precision=settings.openvino.precision
            )
            
            model_info = detection_model.get_model_info()
            logger.info(
                f"[{i}/{len(cameras_ff)}] Modelo de detecção carregado: "
                f"backend={model_info['backend']}, "
                f"device={model_info['device']}, "
                f"precision={model_info['precision']}"
            )
            
            # Cria wrapper YOLOFaceDetector para interface de domínio
            face_detector = YOLOFaceDetector(detection_model)
            
            # POOL GLOBAL de landmarks (carregado uma única vez no início)
            # Não precisa carregar landmarks_detector por câmera
            
            # Cria ProcessFaceDetectionUseCase
            # YOLO gerencia o stream internamente, não precisa de RTSPStreamReader
            processor = ProcessFaceDetectionUseCase(
                camera=camera,
                face_detector=face_detector,
                findface_adapter=findface_adapter,
                findface_queue=findface_queue,  # Fila global compartilhada
                event_creation_service=event_creation_service,
                movement_detection_service=movement_detection_service,
                track_validation_service=track_validation_service,
                track_lifecycle_service=track_lifecycle_service,
                landmarks_detector=landmarks_detector,  # Detector compartilhado (inferência síncrona)
                gpu_id=gpu_id,
                image_save_service=image_save_service,
                face_quality_service=face_quality_service,
                tracker_config=settings.bytetrack.tracker_config,
                show_video=settings.processing.show_video,
                conf_threshold=settings.yolo.conf_threshold,
                iou_threshold=settings.yolo.iou_threshold,
                max_frames_lost=settings.bytetrack.max_frames_lost,
                verbose_log=settings.processing.verbose_log,
                save_images=settings.storage.save_images,
                project_dir=settings.storage.project_dir,
                results_dir=settings.storage.results_dir,
                min_movement_threshold=settings.movement.min_movement_threshold_pixels,
                min_movement_percentage=settings.movement.min_movement_frame_percentage,
                min_confidence_threshold=settings.detection_filter.min_confidence,
                min_bbox_width=settings.detection_filter.min_bbox_width,
                max_frames_per_track=settings.bytetrack.max_frames_per_track,
                inference_size=settings.performance.inference_size,
                detection_skip_frames=settings.performance.detection_skip_frames,
                jpeg_quality=settings.performance.jpeg_compression
            )
            processors.append(processor)
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo para câmera {camera.camera_name.value()}: {e}")
            logger.warning(f"Câmera {camera.camera_name.value()} será ignorada.")
            continue
    
    if not processors:
        logger.error("Nenhuma câmera foi carregada com sucesso. Encerrando.")
        return
    
    logger.info(f"Iniciando {len(processors)} processador(es) de câmera em paralelo...")
    
    # Inicia cada processador em uma thread separada
    threads = []
    try:
        for i, proc in enumerate(processors):
            thread = threading.Thread(
                target=proc.start,
                name=f"Camera-{proc.camera.camera_id.value()}-{proc.camera.camera_name.value()}",
                daemon=True
            )
            thread.start()
            threads.append(thread)
            logger.info(
                f"Thread {i+1}/{len(processors)} iniciada: "
                f"Camera {proc.camera.camera_name.value()} (ID: {proc.camera.camera_id.value()})"
            )
        
        logger.info(f"Todas as {len(threads)} threads de câmera foram iniciadas com sucesso.")
        logger.info("Pressione Ctrl+C para parar o processamento.")
        
        # Mantém o programa principal rodando com timeout para responder ao Ctrl+C
        import time
        while any(thread.is_alive() for thread in threads):
            time.sleep(0.5)  # Verifica a cada 500ms se threads ainda estão vivas
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interrupção detectada (Ctrl+C). Finalizando todas as câmeras...")
        for proc in processors:
            proc.stop()
        
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
        
        # Aguarda threads de câmeras finalizarem (timeout de 5 segundos por thread)
        logger.info("Aguardando threads de câmeras finalizarem...")
        for i, thread in enumerate(threads, 1):
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning(f"Thread {i}/{len(threads)} não finalizou no tempo esperado.")
            else:
                logger.info(f"Thread {i}/{len(threads)} finalizada com sucesso.")
        
        logger.info("✓ Todas as câmeras foram finalizadas.")


if __name__ == "__main__":
    try:
        # Carrega configurações type-safe
        settings = ConfigLoader.load()
        
        # Cria cliente FindFace
        ff = create_findface_client(settings.findface)
        
        # Cria adapter do FindFace com qualidade JPEG configurável
        findface_adapter = FindfaceAdapter(
            ff, 
            camera_prefix=settings.findface.camera_prefix,
            jpeg_quality=settings.performance.jpeg_compression
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