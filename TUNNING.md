🚀 Principais Pontos Fortes de Performance
1. Arquitetura Multi-Thread com Processamento Paralelo
1 thread por câmera - Processamento isolado e paralelo de múltiplos streams RTSP
Pool global de N/2 workers FindFace (mínimo 4) - Distribui envios entre todas as câmeras
1 worker de landmarks por câmera - Inferência assíncrona de keypoints faciais
1 worker global de salvamento de imagens - I/O não bloqueante

2. Filas Assíncronas (Producer-Consumer)
FindFace Queue global (500 itens) - Envios HTTP não bloqueiam detecção
Landmarks Queue por câmera (batch_size × 3) - Inferência em lote sem bloquear tracking
Image Save Queue (200 itens) - Salvamento em disco não bloqueia processamento
Log Queue (10.000 itens) - Logging não bloqueante com QueueHandler

3. Processamento em Lote (Batch Processing)
GPU batch inference: gpu_batch_size: 32 - Maximiza throughput da GPU
Landmarks em batch: Acumula múltiplas faces e infere todas de uma vez
Timeout inteligente: Não espera batch completo - processa parcial após 100ms

4. Otimizações de Memória
Zero-copy frames: FullFrameVO(copy=False) - Economia de ~70% de RAM
Referência read-only: Evita duplicação de arrays numpy
Garbage collection periódica: A cada 500 tracks finalizados para liberar memória

5. Skip Frames Configurável
detection_skip_frames: 1 - Processa 1 a cada N frames
ByteTrack continua tracking mesmo em frames pulados
Trade-off latência/throughput: Menos frames = maior FPS total

6. Inferência Otimizada
inference_size: 640 - Resolução menor = 4x mais rápido que 1280px
FP16 em GPU: Half-precision para GPUs NVIDIA (2x mais rápido)
TensorRT/OpenVINO: Backends otimizados quando disponíveis
Streaming mode: YOLO .track(stream=True) - Generator sem buffer

7. Seleção Inteligente da Melhor Face
Score de qualidade composto:

Frontalidade (peso 6) - Face de frente tem prioridade
Tamanho do bbox (peso 4) - Faces maiores são melhores
Confiança YOLO (peso 3) - Detecções mais confiáveis
Nitidez (peso 1) - Faces nítidas preferidas
Proporção (peso 1) - Bboxes bem proporcionados
Apenas 1 evento por track enviado ao FindFace (melhor qualidade)

Reduz tráfego de rede drasticamente

8. Filtros de Validação Pré-Envio
Movimento mínimo: 0.7 pixels - Elimina faces estáticas/falsas
Confiança mínima: 0.40 - Descarta detecções duvidosas
Largura mínima bbox: 35 pixels - Ignora faces muito pequenas
Percentual de movimento: 10% dos frames - Track deve ter movimento real

9. I/O Não Bloqueante
Salvamento assíncrono: Worker dedicado para cv2.imwrite()
Envios HTTP assíncronos: Pool de workers para API FindFace
Logging assíncrono: QueueListener processa logs em background
Thread principal 100% dedicada à detecção/tracking

10. Configuração Dinâmica de Workers
num_cpus = multiprocessing.cpu_count()
num_findface_workers = max(4, num_cpus // 2)  # Escala com hardware

11. Gestão Eficiente de Tracks
ByteTrack nativo: Tracking stateful sem overhead adicional
Kalman filter: Predição de posições entre frames
Max frames lost: 60 - Mantém tracks temporariamente ocultas
Max frames por track: 900 - Evita tracks infinitos

12. Redução de Logs no Loop Principal
# OTIMIZAÇÃO: Log verbose removido do loop para evitar gargalo de I/O
# Apenas logs essenciais em DEBUG level

┌─────────────────────────────────────────────────────────────────┐
│ Frame RTSP → YOLO detect → ByteTrack → Validation → Queue      │
│   ~1ms        ~15-30ms      ~2ms         ~0.5ms      ~0.1ms    │
└─────────────────────────────────────────────────────────────────┘
                                ↓ (assíncrono)
┌─────────────────────────────────────────────────────────────────┐
│ Landmarks (batch) → Quality Score → Best Face → FindFace Queue │
│   ~10-20ms/batch      ~1ms           ~0.1ms       ~0.1ms       │
└─────────────────────────────────────────────────────────────────┘
                                ↓ (worker pool)
┌─────────────────────────────────────────────────────────────────┐
│ FindFace HTTP POST → Resposta                                  │
│   ~50-200ms (rede)                                             │
└─────────────────────────────────────────────────────────────────┘