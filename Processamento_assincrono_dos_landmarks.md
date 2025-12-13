A inferência em lote de landmarks utiliza um padrão Producer-Consumer assíncrono com as seguintes características:

Thread principal (producer) - Enfileira crops de faces
Worker thread (consumer) - Acumula crops em batches e processa
Cache de resultados - Armazena landmarks processados
Timeout inteligente - Não espera batch completo

🏗️ Estrutura de Dados
1. Inicialização do Sistema de Batch
==========================================================
# Cache de resultados: event_id -> landmarks array
self._landmarks_results: Dict[int, Optional[np.ndarray]] = {}

# Tamanho do batch = gpu_batch_size (32) ou cpu_batch_size (1)
self._landmarks_batch_size = batch  

if self.landmarks_model is not None:
    # Fila: 3x o batch size (buffer para evitar bloqueio)
    # Exemplo: batch=32 → fila de 96 itens
    landmarks_queue_size = self._landmarks_batch_size * 3
    self._landmarks_queue = Queue(maxsize=landmarks_queue_size)
    
    # Inicia worker thread em background
    self._landmarks_worker_running = True
    self._landmarks_worker = Thread(
        target=self._landmarks_batch_worker,
        name=f"Landmarks-Worker-{camera.camera_id.value()}",
        daemon=True
    )
    self._landmarks_worker.start()
==========================================================

┌─────────────────────────────────────────────────────┐
│ Queue: [(event_id, face_crop), ...]                │
│ Capacidade: batch_size × 3 (ex: 32 × 3 = 96)       │
└─────────────────────────────────────────────────────┘
         ↓ (producer enfileira)
┌─────────────────────────────────────────────────────┐
│ Worker Thread: Acumula em batch                    │
│ Buffer: [(1, crop1), (2, crop2), ..., (32, crop32)]│
└─────────────────────────────────────────────────────┘
         ↓ (processa batch)
┌─────────────────────────────────────────────────────┐
│ Cache: {1: landmarks1, 2: landmarks2, ...}         │
└─────────────────────────────────────────────────────┘

📤 Producer: Enfileirando Crops de Faces
2. Thread Principal Enfileira Crops

==========================================================
def _create_event(self, frame, box, keypoints, index):
    self._event_id_counter += 1
    event_id = self._event_id_counter
    
    # Extrai bbox e confiança
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    bbox = BboxVO((x1, y1, x2, y2))
    confidence = ConfidenceVO(float(box.conf[0]))
    
    # ═══════════════════════════════════════════════════
    # PASSO 1: ENFILEIRA CROP PARA PROCESSAMENTO ASSÍNCRONO
    # ═══════════════════════════════════════════════════
    landmarks_array = None
    if self.landmarks_model is not None and self._landmarks_queue is not None:
        try:
            # Obtém frame original (zero-copy)
            frame_array = frame.full_frame.ndarray_readonly
            
            # Faz crop da face usando bbox
            face_crop = frame_array[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
            
            # Enfileira para processamento em lote
            if face_crop.size > 0:
                try:
                    # put_nowait = não bloqueia se fila cheia
                    self._landmarks_queue.put_nowait((event_id, face_crop.copy()))
                except:
                    # Fila cheia - continua sem landmarks (fallback)
                    pass
            
            # ═══════════════════════════════════════════════════
            # PASSO 2: TENTA BUSCAR RESULTADO JÁ PROCESSADO
            # ═══════════════════════════════════════════════════
            # pop = remove do cache se existir
            landmarks_array = self._landmarks_results.pop(event_id, None)
                
        except Exception:
            # Erro silencioso - continua sem landmarks
            pass
    
    # Fallback: usa keypoints do YOLO de detecção
    if landmarks_array is None and keypoints is not None:
        kpts = keypoints[index].xy[0].cpu().numpy()
        landmarks_array = kpts
    
    landmarks = LandmarksVO(landmarks_array)
    
    # Cria evento com landmarks (ou None se não disponível)
    event = Event(
        id=IdVO(event_id),
        frame=frame,
        bbox=bbox,
        confidence=confidence,
        landmarks=landmarks
    )
    
    return event
==========================================================	

Fluxo:
Frame → YOLO detecta 5 faces
   ↓
Para cada face:
   ├─ Extrai bbox (x1, y1, x2, y2)
   ├─ Faz crop: face_crop = frame[y1:y2, x1:x2]
   ├─ Enfileira: queue.put((event_id, face_crop))
   └─ Busca resultado (se já processado): cache.pop(event_id)
   
📥 Consumer: Worker de Batch Processing
3. Worker Acumula e Processa Batches   
==========================================================	

def _landmarks_batch_worker(self):
    """Worker thread dedicado para inferência de landmarks em lote."""
    self.logger.info("Landmarks worker iniciado")
    
    batch_buffer = []  # Buffer temporário: [(event_id, face_crop), ...]
    
    while self._landmarks_worker_running:
        try:
            # ═══════════════════════════════════════════════════
            # FASE 1: PEGA PRIMEIRO ITEM (BLOQUEIA ATÉ 0.1s)
            # ═══════════════════════════════════════════════════
            try:
                item = self._landmarks_queue.get(timeout=0.1)
                
                if item is None:  # Sinal de parada
                    self.logger.info("Landmarks worker recebeu sinal de parada")
                    break
                
                batch_buffer.append(item)
            except:
                # Timeout - continua sem item
                if not batch_buffer:
                    continue
            
            # ═══════════════════════════════════════════════════
            # FASE 2: ACUMULA ATÉ COMPLETAR BATCH (TIMEOUT 10ms)
            # ═══════════════════════════════════════════════════
            while len(batch_buffer) < self._landmarks_batch_size:
                try:
                    # Timeout CURTO (10ms) para não esperar muito
                    item = self._landmarks_queue.get(timeout=0.01)
                    if item is None:
                        break
                    batch_buffer.append(item)
                except:
                    break  # Timeout - processa o que tem
            
            if not batch_buffer:
                continue
            
            # ═══════════════════════════════════════════════════
            # FASE 3: PROCESSA BATCH
            # ═══════════════════════════════════════════════════
            try:
                # Separa IDs e crops
                event_ids = [item[0] for item in batch_buffer]
                crops = [item[1] for item in batch_buffer]
                
                # INFERE LANDMARKS EM LOTE (um por vez, mas agrupado)
                for event_id, crop in zip(event_ids, crops):
                    try:
                        if crop.size > 0:
                            # Chama modelo YOLO de landmarks
                            landmarks_result = self.landmarks_model.predict(
                                face_crop=crop,
                                conf=self.conf,
                                verbose=False  # Sem logs em batch
                            )
                            
                            if landmarks_result is not None:
                                landmarks_array, _ = landmarks_result
                                # Armazena no cache
                                self._landmarks_results[event_id] = landmarks_array
                            else:
                                self._landmarks_results[event_id] = None
                        else:
                            self._landmarks_results[event_id] = None
                    except Exception as e:
                        self._landmarks_results[event_id] = None
                
                # Marca tarefas como concluídas
                for _ in batch_buffer:
                    self._landmarks_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Erro no processamento de batch: {e}")
                # Marca como concluídas mesmo com erro
                for _ in batch_buffer:
                    try:
                        self._landmarks_queue.task_done()
                    except:
                        pass
            
            # Limpa buffer para próximo batch
            batch_buffer.clear()
            
        except Exception as e:
            if not self._landmarks_worker_running:
                break
            self.logger.error(f"Erro no landmarks worker: {e}")
            continue
    
    self.logger.info("Landmarks worker finalizado")
==========================================================	

⚡ Exemplo Prático de Fluxo Completo
Cenário: 5 faces detectadas em 1 frame

T=0ms - Frame capturado, YOLO detecta 5 faces
  │
  ├─ Face 1: enfileira (event_id=1, crop1) → Queue: [1]
  ├─ Face 2: enfileira (event_id=2, crop2) → Queue: [1, 2]
  ├─ Face 3: enfileira (event_id=3, crop3) → Queue: [1, 2, 3]
  ├─ Face 4: enfileira (event_id=4, crop4) → Queue: [1, 2, 3, 4]
  └─ Face 5: enfileira (event_id=5, crop5) → Queue: [1, 2, 3, 4, 5]
  
T=5ms - Worker acorda e começa a acumular batch
  │
  ├─ get() → pega item 1 → buffer: [1]
  ├─ get(timeout=0.01) → pega item 2 → buffer: [1, 2]
  ├─ get(timeout=0.01) → pega item 3 → buffer: [1, 2, 3]
  ├─ get(timeout=0.01) → pega item 4 → buffer: [1, 2, 3, 4]
  └─ get(timeout=0.01) → pega item 5 → buffer: [1, 2, 3, 4, 5]
  
T=10ms - Processa batch de 5 faces
  │
  ├─ crop1 → landmarks_model.predict() → landmarks1
  ├─ crop2 → landmarks_model.predict() → landmarks2
  ├─ crop3 → landmarks_model.predict() → landmarks3
  ├─ crop4 → landmarks_model.predict() → landmarks4
  └─ crop5 → landmarks_model.predict() → landmarks5
  
T=25ms - Armazena resultados no cache
  │
  └─ cache = {1: landmarks1, 2: landmarks2, ..., 5: landmarks5}

T=30ms - Próximo frame detecta 3 faces (IDs: 6, 7, 8)
  │
  ├─ event_id=6: enfileira crop → busca cache (não existe ainda)
  ├─ event_id=7: enfileira crop → busca cache (não existe ainda)
  └─ event_id=8: enfileira crop → busca cache (não existe ainda)

T=35ms - Worker processa batch de 3
  
T=50ms - Frame anterior (com 5 faces) finaliza track
  │
  └─ Seleciona melhor face: event_id=3 tem landmarks3 do cache ✅