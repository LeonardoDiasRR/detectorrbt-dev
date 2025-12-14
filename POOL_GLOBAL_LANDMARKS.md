# Pool Global de Landmarks - Documentação

## Descrição

Implementação de pool global de workers para inferência de landmarks faciais, seguindo o mesmo padrão bem-sucedido do pool FindFace.

## Motivação

### Problema Anterior (1 worker por câmera)
- **Ineficiência**: Câmera com muitas faces sobrecarga seu worker
- **Desperdício**: Câmera com poucas faces deixa worker ocioso
- **Sem Load Balancing**: Workers não compartilham carga entre câmeras
- **Memória**: N câmeras × 1 modelo = N instâncias do modelo landmarks

### Solução Atual (Pool Global)
- **Eficiência**: Workers processam de QUALQUER câmera
- **Load Balancing**: Carga distribuída automaticamente
- **Economia de Memória**: 1 modelo único compartilhado
- **Escalabilidade**: max(4, CPUs // 2) workers otimizados

## Arquitetura

### Componentes

#### 1. Pool Global (run.py)
```python
# Fila global compartilhada
landmarks_queue = Queue(maxsize=1000)

# Cache global: {camera_id: {event_id: landmarks}}
landmarks_cache = {}
landmarks_cache_lock = threading.Lock()

# Workers: max(4, num_cpus // 2)
num_landmarks_workers = max(4, num_cpus // 2)

# Detector único compartilhado
landmarks_detector = YOLOLandmarksDetector(
    model_path=settings.model.landmarks_model_path,
    device=settings.model.device
)
```

#### 2. Worker Function (run.py)
```python
def landmarks_worker_func(worker_id, queue, detector, cache, lock, batch_size, running_flag):
    """
    Worker que processa crops de TODAS as câmeras em batch.
    
    FASE 1: Pega primeiro item (timeout 100ms)
    FASE 2: Acumula batch de qualquer câmera (timeout 10ms)
    FASE 3: Batch inference
    FASE 4: Armazena no cache global (thread-safe)
    """
```

#### 3. Câmeras (ProcessFaceDetectionUseCase)
```python
# Recebe recursos globais
self._landmarks_queue = landmarks_queue  # Fila compartilhada
self._landmarks_results_cache = landmarks_cache  # Cache global
self._landmarks_cache_lock = landmarks_cache_lock  # Lock

# Enfileira para pool global
self._landmarks_queue.put_nowait((camera_id, event_id, face_crop.copy()))

# Recupera do cache global (thread-safe)
with self._landmarks_cache_lock:
    camera_cache = self._landmarks_results_cache.get(camera_id, {})
    landmarks_array = camera_cache.pop(event_id, None)
```

## Formato de Dados

### Fila de Entrada
```python
# Item: (camera_id, event_id, face_crop)
item = (1, 42, np.array([...]))  # Câmera 1, Evento 42, Crop da face
```

### Cache de Saída
```python
# Cache: {camera_id: {event_id: landmarks_array}}
cache = {
    1: {42: np.array([[x1,y1], [x2,y2], ...])},  # Câmera 1
    2: {15: np.array([[x1,y1], [x2,y2], ...])},  # Câmera 2
}
```

## Configuração

### Parâmetros do Pool
- **Workers**: max(4, num_cpus // 2)
- **Fila**: maxsize=1000 (maior que FindFace)
- **Batch Size GPU**: 32 (configurável via `gpu_batch_size`)
- **Batch Size CPU**: 1 (configurável via `cpu_batch_size`)

### Timeouts
- **Primeiro item**: 100ms (timeout do get inicial)
- **Itens batch**: 10ms por item (acumulação rápida)
- **Total batch**: ~100-200ms (processamento completo)

## Thread Safety

### Lock Usage
```python
# SEMPRE usar lock ao acessar cache global
with landmarks_cache_lock:
    if camera_id not in cache:
        cache[camera_id] = {}
    cache[camera_id][event_id] = landmarks_array
```

### Queue Safety
- `queue.put_nowait()`: Não bloqueia se fila cheia (fallback)
- `queue.get(timeout=0.1)`: Timeout curto para responsividade
- `queue.task_done()`: Sempre chamar após processar

## Benefícios Esperados

### Performance
- **+50-80% utilização** de workers (vs. per-camera)
- **-30-50ms latência** em cenários de carga desbalanceada
- **+2-3× throughput** quando câmeras têm carga muito variável

### Recursos
- **-70-90% memória** de modelos landmarks (1 vs. N instâncias)
- **Load balancing automático** entre câmeras
- **Escalabilidade horizontal** com CPUs

### Manutenção
- **Código centralizado**: Pool em run.py
- **Debugging simplificado**: Logs por worker, não por câmera
- **Configuração única**: Batch size global

## Comparação com FindFace Pool

| Aspecto | FindFace Pool | Landmarks Pool |
|---------|---------------|----------------|
| **Workers** | max(4, N/2) | max(4, N/2) ✓ |
| **Fila** | Queue(500) | Queue(1000) ✓ |
| **Cache** | Global dict | Global dict ✓ |
| **Lock** | threading.Lock | threading.Lock ✓ |
| **Batch** | Não (API REST) | Sim (batch_size) ✓ |
| **Formato** | (cam_id, cam_name, track_id, event, count) | (cam_id, event_id, crop) ✓ |

## Migração

### Antes (Per-Camera)
```python
# Em ProcessFaceDetectionUseCase.__init__
self._landmarks_queue = Queue(maxsize=batch_size * 3)
self._landmarks_worker = threading.Thread(
    target=self._landmarks_batch_worker,
    name=f"Landmarks-Worker-{camera_id}",
    daemon=True
)
self._landmarks_worker.start()
```

### Depois (Global Pool)
```python
# Em run.py
landmarks_queue = Queue(maxsize=1000)
landmarks_cache = {}
landmarks_cache_lock = threading.Lock()

# Workers iniciados ANTES do loop de câmeras
for i in range(num_landmarks_workers):
    worker = threading.Thread(
        target=landmarks_worker_func,
        args=(i+1, queue, detector, cache, lock, batch_size, running_flag),
        daemon=True
    )
    worker.start()

# Em ProcessFaceDetectionUseCase.__init__
self._landmarks_queue = landmarks_queue  # Recebe global
self._landmarks_cache = landmarks_cache  # Recebe global
self._landmarks_lock = landmarks_cache_lock  # Recebe global
```

## Testes Recomendados

### Cenário 1: Carga Balanceada
- **Setup**: 4 câmeras, ~10 faces cada
- **Expectativa**: 90%+ utilização de workers
- **Métrica**: Latência média < 100ms

### Cenário 2: Carga Desbalanceada
- **Setup**: 1 câmera 50 faces, 3 câmeras 5 faces
- **Expectativa**: Workers processam câmera carregada em paralelo
- **Métrica**: 2-3× throughput vs. per-camera

### Cenário 3: Stress Test
- **Setup**: 8 câmeras, 100 faces simultâneas
- **Expectativa**: Fila não enche (< 1000 itens)
- **Métrica**: 0% perda de crops

## Troubleshooting

### Problema: Fila cheia
```python
# Sintoma: Logs "Fila cheia - usando fallback"
# Solução: Aumentar landmarks_queue maxsize OU aumentar workers
landmarks_queue = Queue(maxsize=2000)  # Era 1000
num_landmarks_workers = max(8, num_cpus)  # Era max(4, num_cpus // 2)
```

### Problema: Cache crescendo
```python
# Sintoma: Memória aumentando indefinidamente
# Causa: Evento enfileirado mas nunca recuperado
# Solução: Limpar cache periodicamente no stop()
with self._landmarks_cache_lock:
    if camera_id in self._landmarks_results_cache:
        del self._landmarks_results_cache[camera_id]
```

### Problema: Latência alta
```python
# Sintoma: Latência > 200ms consistentemente
# Causa: Batch size muito grande OU workers insuficientes
# Solução: Reduzir batch OU aumentar workers
gpu_batch_size = 16  # Era 32
num_landmarks_workers = max(8, num_cpus)  # Era max(4, num_cpus // 2)
```

## Monitoramento

### Logs Importantes
```
INFO - Criando pool global de Landmarks com N workers (batch: B, dispositivo: GPU/CPU)
INFO - Pool de N workers Landmarks iniciado
INFO - Detector de landmarks carregado: yolov8n-face.pt
INFO - Worker N iniciado (batch_size: B)
INFO - Usando pool GLOBAL de landmarks (fila compartilhada entre câmeras)
```

### Métricas a Observar
- **Queue size**: landmarks_queue.qsize() < maxsize * 0.8
- **Cache size**: len(landmarks_cache) ≈ num_cameras
- **Worker activity**: Logs de "Worker N iniciado/finalizado"
- **Throughput**: Eventos processados por segundo

## Referências

- **Arquivo**: [run.py](run.py) - Linhas 245-410 (pool global)
- **Arquivo**: [process_face_detection_use_case.py](src/application/use_cases/process_face_detection_use_case.py) - Linhas 120-170 (uso do pool)
- **Inspiração**: FindFace Pool (run.py, linhas 168-245)
- **Documento**: [Processamento_assincrono_dos_landmarks.md](Processamento_assincrono_dos_landmarks.md)
