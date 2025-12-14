# Configuração Multi-GPU

## Visão Geral

O DetectoRR agora suporta distribuição automática de câmeras entre múltiplas GPUs usando estratégia **round-robin**. Esta funcionalidade permite escalar horizontalmente o processamento para ambientes com dezenas de câmeras.

## Como Funciona

### Distribuição Round-Robin

Cada câmera é atribuída sequencialmente a uma GPU da lista configurada:

- **Câmera 1** → GPU 0
- **Câmera 2** → GPU 1  
- **Câmera 3** → GPU 2
- **Câmera 4** → GPU 3
- **Câmera 5** → GPU 0 (ciclo reinicia)
- **Câmera 6** → GPU 1
- ...e assim por diante

### Vantagens

✅ **Simples**: Configuração direta via arquivo YAML  
✅ **Automático**: Distribuição feita automaticamente pelo sistema  
✅ **Balanceado**: Carga uniforme entre GPUs (± 1 câmera de diferença)  
✅ **Escalável**: Adicione mais GPUs conforme necessário  
✅ **Sem overhead**: Nenhuma sincronização inter-GPU necessária

## Configuração

### 1. Arquivo `config.yaml`

```yaml
# Lista de GPUs a serem utilizadas (distribuição round-robin entre câmeras)
# Exemplos: 
#   [0]          - usa apenas GPU 0
#   [0, 1]       - usa GPU 0 e 1
#   [0, 1, 2, 3] - usa 4 GPUs
gpu_devices: [0, 1, 2, 3]
```

### 2. Verificar GPUs Disponíveis

Execute no terminal:

```bash
nvidia-smi --list-gpus
```

Exemplo de saída:
```
GPU 0: NVIDIA RTX A4000 (UUID: GPU-xxx)
GPU 1: NVIDIA RTX A4000 (UUID: GPU-yyy)
GPU 2: NVIDIA RTX A4000 (UUID: GPU-zzz)
GPU 3: NVIDIA RTX A4000 (UUID: GPU-www)
```

### 3. Logs da Aplicação

Ao iniciar, você verá logs indicando a distribuição:

```
INFO - Distribuindo 20 câmera(s) entre 4 GPU(s): [0, 1, 2, 3]
INFO - [1/20] Câmera CAM_ENTRADA → GPU 0
INFO - [1/20] GPU 0 selecionada para câmera CAM_ENTRADA
INFO - [2/20] Câmera CAM_SAIDA → GPU 1
INFO - [2/20] GPU 1 selecionada para câmera CAM_SAIDA
...
```

## Exemplos de Cenários

### Cenário 1: 20 Câmeras, 1 GPU

```yaml
gpu_devices: [0]
```

**Carga por GPU**: 20 câmeras × 30 FPS = 600 frames/s  
**Status**: ⚠️ **Sobrecarga** (RTX A4000 processa ~50 frames/s)

### Cenário 2: 20 Câmeras, 2 GPUs

```yaml
gpu_devices: [0, 1]
```

**Carga por GPU**: 10 câmeras × 30 FPS = 300 frames/s  
**Status**: ⚠️ **Ainda sobrecarregado** (6× capacidade)

### Cenário 3: 20 Câmeras, 4 GPUs ✅ (RECOMENDADO)

```yaml
gpu_devices: [0, 1, 2, 3]

performance:
  detection_skip_frames: 2  # Processa 1 a cada 2 frames
```

**Carga por GPU**: 5 câmeras × 15 FPS efetivos = 75 frames/s  
**Status**: ✅ **Balanceado** (1.5× capacidade, mas gerenciável)

### Cenário 4: 40 Câmeras, 4 GPUs

```yaml
gpu_devices: [0, 1, 2, 3]

performance:
  detection_skip_frames: 3  # Processa 1 a cada 3 frames
```

**Carga por GPU**: 10 câmeras × 10 FPS efetivos = 100 frames/s  
**Status**: ⚠️ **Próximo ao limite** (2× capacidade)

## Cálculo de Capacidade

### Capacidade por GPU (RTX A4000 com TensorRT)

- **Modelo YOLO + TensorRT FP16**: ~50-66 frames/s
- **Resolução reduzida (inference_size: 640)**: +20-30% performance
- **Frame skipping (detection_skip_frames: 2)**: Divide carga por 2

### Fórmula de Carga

```
Carga por GPU = (Número de câmeras ÷ Número de GPUs) × (FPS ÷ detection_skip_frames)
```

**Exemplo com 20 câmeras, 4 GPUs, skip=2:**
```
Carga = (20 ÷ 4) × (30 ÷ 2) = 5 × 15 = 75 frames/s por GPU
```

## Parâmetros Relacionados

Combine multi-GPU com outros parâmetros de performance para máxima eficiência:

```yaml
gpu_devices: [0, 1, 2, 3]

performance:
  # Reduz resolução de inferência (2-4× mais rápido)
  inference_size: 640
  
  # Processa 1 a cada N frames (reduz carga proporcionalmente)
  detection_skip_frames: 2
  
tensorrt:
  enabled: true
  precision: FP16  # Essencial para performance
  workspace: 4
```

## Monitoramento

### Durante Execução

Monitore uso das GPUs em tempo real:

```bash
watch -n 1 nvidia-smi
```

Você verá algo assim:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA 12.2  |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA RTX A4000    On   | 00000000:01:00.0 Off |                  Off |
| 41%   62C    P2   128W / 140W |  12456MiB / 16376MiB |     85%      Default |
|   1  NVIDIA RTX A4000    On   | 00000000:02:00.0 Off |                  Off |
| 43%   64C    P2   132W / 140W |  12678MiB / 16376MiB |     88%      Default |
|   2  NVIDIA RTX A4000    On   | 00000000:03:00.0 Off |                  Off |
| 40%   61C    P2   125W / 140W |  12234MiB / 16376MiB |     82%      Default |
|   3  NVIDIA RTX A4000    On   | 00000000:04:00.0 Off |                  Off |
| 42%   63C    P2   129W / 140W |  12543MiB / 16376MiB |     86%      Default |
+-------------------------------+----------------------+----------------------+
```

**Indicadores saudáveis:**
- ✅ GPU-Util entre 70-90% (balanceado)
- ✅ Memory-Usage < 90% (sem overflow)
- ✅ Temperatura < 80°C (resfriamento adequado)

## Troubleshooting

### Problema: Erro "CUDA out of memory"

**Causa**: Modelo muito grande ou muitas câmeras por GPU

**Soluções**:
1. Adicione mais GPUs ao `gpu_devices`
2. Aumente `detection_skip_frames` (reduz frames processados)
3. Reduza `inference_size` (modelos menores)
4. Reduza `gpu_batch_size`

### Problema: GPUs desequilibradas

**Causa**: Número de câmeras não é múltiplo do número de GPUs

**Exemplo**: 21 câmeras, 4 GPUs
- GPU 0: 6 câmeras
- GPU 1: 5 câmeras  
- GPU 2: 5 câmeras
- GPU 3: 5 câmeras

**Solução**: Isso é normal e esperado (diferença máxima de 1 câmera)

### Problema: "RuntimeError: cuda runtime error (10)"

**Causa**: ID de GPU inválido em `gpu_devices`

**Solução**: Verifique IDs disponíveis com `nvidia-smi --list-gpus`

### Problema: Uma GPU não está sendo usada

**Causa**: Possíveis problemas:
1. GPU desabilitada no sistema
2. Driver NVIDIA não carregado
3. ID incorreto em `gpu_devices`

**Verificações**:
```bash
# Lista GPUs visíveis
nvidia-smi --list-gpus

# Verifica CUDA disponível
python -c "import torch; print(torch.cuda.device_count())"
```

## Migração de Configuração Antiga

### Antes (gpu_index)

```yaml
gpu_index: 0
```

### Depois (gpu_devices)

```yaml
# Usa apenas GPU 0 (comportamento idêntico)
gpu_devices: [0]

# OU usa múltiplas GPUs
gpu_devices: [0, 1, 2, 3]
```

**Nota**: O sistema ainda suporta `gpu_index` para compatibilidade, mas `gpu_devices` tem prioridade.

## Performance Esperada

### 20 Câmeras, 4 GPUs, RTX A4000

**Configuração**:
```yaml
gpu_devices: [0, 1, 2, 3]
performance:
  inference_size: 640
  detection_skip_frames: 2
  max_parallel_workers: 10
tensorrt:
  enabled: true
  precision: FP16
```

**Resultados Esperados**:
- 📊 FPS por câmera: **25-28 FPS**
- ⏱️ Latência média: **400-600ms**
- 🔥 Uso por GPU: **85-90%**
- 💾 VRAM por GPU: **11-13GB**
- 🎯 Taxa de detecção: **>95%**

## Recomendações

1. **Use TensorRT**: Essencial para performance (~2-3× speedup)
2. **Adicione GPUs conforme necessário**: 1 GPU a cada 5-7 câmeras
3. **Monitore uso**: `nvidia-smi` deve mostrar 70-90% utilização
4. **Ajuste skip_frames**: Balance entre FPS e carga
5. **Teste incrementalmente**: Adicione câmeras gradualmente

## Suporte

Para dúvidas sobre configuração multi-GPU, consulte também:
- `PERFORMANCE.md` - Guia completo de otimização
- Logs da aplicação em `detectorrbt.log`
- Output de `nvidia-smi` para diagnóstico
