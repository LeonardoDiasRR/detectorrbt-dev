# Parâmetros de Configuração - Status de Implementação

## Resumo
✅ **25 de 25** parâmetros do `config.yaml` estão sendo efetivamente utilizados na aplicação.

---

## Parâmetros Implementados

### 1. **Configurações de Câmera** (`cameras`)
- ✅ `nome`: Nome da câmera → `Camera.camera_name`
- ✅ `url`: URL do stream RTSP → `Camera.camera_source`
- ✅ `token`: Token do FindFace → `Camera.camera_token`

**Fluxo**: `config.yaml` → `ConfigLoader` → `AppSettings.cameras` → `run.py` → `Camera entity`

---

### 2. **Configurações de Processamento** (`processamento`)
- ✅ `conf_threshold`: Threshold de confiança → `ProcessFaceDetectionUseCase.conf_threshold`
- ✅ `iou_threshold`: Threshold de IoU para NMS → `ProcessFaceDetectionUseCase.iou_threshold`
- ✅ `inference_size`: Tamanho de inferência → `ProcessFaceDetectionUseCase.inference_size`
- ✅ `detection_skip_frames`: Frames a pular → `ProcessFaceDetectionUseCase.detection_skip_frames`
- ✅ `cpu_batch_size`: Batch size CPU → `YOLOFaceDetector.detect_and_track(batch=...)`
- ✅ `gpu_batch_size`: Batch size GPU → `YOLOFaceDetector.detect_and_track(batch=...)`
- ✅ `show`: Exibir vídeo → `YOLOFaceDetector.detect_and_track(show=...)`

**Fluxo**: 
- `config.yaml` → `ConfigLoader` → `AppSettings.processing` → `run.py` → `ProcessFaceDetectionUseCase`
- `batch_size` e `show` propagados para `IFaceDetector.detect_and_track()` → `YOLOFaceDetector` → `model.track()`

**Última implementação**: Parâmetros `batch` e `show` adicionados à interface `IFaceDetector` e propagados através de toda a cadeia de detecção.

---

### 3. **Configurações de Tracking** (`tracking`)
- ✅ `max_frames_lost`: Máximo de frames perdidos → `ProcessFaceDetectionUseCase.max_frames_lost`

**Fluxo**: `config.yaml` → `ConfigLoader` → `AppSettings.tracking` → `run.py` → `ProcessFaceDetectionUseCase`

---

### 4. **Configurações de Validação** (`validacao`)
- ✅ `min_movement_threshold`: Distância mínima de movimento → `TrackValidationService.min_movement_threshold`
- ✅ `min_movement_percentage`: Percentual mínimo de movimento → `TrackValidationService.min_movement_percentage`
- ✅ `min_confidence_threshold`: Confiança mínima → `TrackValidationService.min_confidence_threshold`
- ✅ `min_bbox_width`: Largura mínima do bbox → `TrackValidationService.min_bbox_width`
- ✅ `max_frames_per_track`: Máximo de frames por track → `TrackLifecycleService.max_frames_per_track`

**Fluxo**: `config.yaml` → `ConfigLoader` → `AppSettings.validation` → `run.py` → `TrackValidationService` e `TrackLifecycleService`

---

### 5. **Configurações de Qualidade Facial** (`qualidade_face`)
- ✅ `peso_confianca`: Peso da confiança → `EventCreationService` → `FaceQualityService.calculate_quality()`
- ✅ `peso_tamanho`: Peso do tamanho → `EventCreationService` → `FaceQualityService.calculate_quality()`
- ✅ `peso_frontal`: Peso da frontalidade → `EventCreationService` → `FaceQualityService.calculate_quality()`
- ✅ `peso_proporcao`: Peso da proporção → `EventCreationService` → `FaceQualityService.calculate_quality()`
- ✅ `peso_nitidez`: Peso da nitidez → `EventCreationService` → `FaceQualityService.calculate_quality()`

**Fluxo**: `config.yaml` → `ConfigLoader` → `AppSettings.face_quality: FaceQualityConfig` → `run.py` → `EventCreationService(peso_*)` → `FaceQualityService`

---

### 6. **Configurações de Modelo** (`modelo`)
- ✅ `modelo_path`: Caminho do modelo YOLO → `ModelFactory.create_model()`
- ✅ `backend`: Backend de inferência (tensorrt/openvino/yolo) → `ModelFactory.create_model()`
- ✅ `device`: Dispositivo (cuda/cpu) → `model_class(device=...)`

**Fluxo**: `config.yaml` → `ConfigLoader` → `AppSettings.model` → `run.py` → `ModelFactory.create_model()` → `YOLOModelAdapter/TensorRTAdapter/OpenVINOAdapter`

---

### 7. **Configurações de Log** (`log`)
- ✅ `verbose_log`: Log detalhado → `ProcessFaceDetectionUseCase.verbose_log`

**Fluxo**: `config.yaml` → `ConfigLoader` → `AppSettings.log` → `run.py` → `ProcessFaceDetectionUseCase`

---

### 8. **Configurações de Salvamento de Imagens** (`save_images`)
- ✅ `enabled`: Habilitar salvamento → `ProcessFaceDetectionUseCase.save_images`

**Fluxo**: `config.yaml` → `ConfigLoader` → `AppSettings.save_images_config` → `run.py` → `ProcessFaceDetectionUseCase`

---

## Arquivos Modificados na Última Implementação

### 1. `src/domain/interfaces/face_detector.py`
**Alterações**:
- Adicionado parâmetro `batch: int = 1` ao método `detect_and_track()`
- Adicionado parâmetro `show: bool = False` ao método `detect_and_track()`

**Motivo**: Definir o contrato da interface para que implementações aceitem batch size e exibição de vídeo.

---

### 2. `src/infrastructure/ml/yolo_face_detector.py`
**Alterações**:
- Adicionado parâmetro `batch: int = 1` ao método `detect_and_track()`
- Adicionado parâmetro `show: bool = False` ao método `detect_and_track()`
- Adicionado `'batch': batch` ao dicionário `track_params`
- Adicionado `'show': show` ao dicionário `track_params`

**Motivo**: Propagar batch size e show para o método `model.track()` do YOLO/Ultralytics.

---

### 3. `src/application/use_cases/process_face_detection_use_case.py`
**Alterações**:
- Adicionado `self.batch_size = batch_size` no `__init__`
- Adicionado `batch=self.batch_size` ao chamar `face_detector.detect_and_track()`
- Adicionado `show=self.show_video` ao chamar `face_detector.detect_and_track()`

**Motivo**: Armazenar e propagar batch size e show video para o detector.

---

## Verificação Final

### Comando para testar show=True
```yaml
# Em config.yaml
processamento:
  show: true  # Habilita exibição de vídeo em tempo real
```

Ao executar `python run.py`, o vídeo será exibido em uma janela OpenCV (via `cv2.imshow()` chamado internamente pelo YOLO quando `show=True`).

### Comando para testar batch_size
```yaml
# Em config.yaml
processamento:
  cpu_batch_size: 16  # Para CPU
  gpu_batch_size: 32  # Para GPU (CUDA)
```

O batch size será propagado para `model.track(batch=16)` ou `model.track(batch=32)` dependendo do dispositivo.

---

## Conclusão

✅ Todos os 25 parâmetros de `config.yaml` estão sendo efetivamente utilizados.

✅ Arquitetura DDD completamente funcional com:
- Domain Layer (interfaces, services, entities, value objects)
- Application Layer (use cases)
- Infrastructure Layer (implementations, adapters, factories)

✅ Sistema de configuração robusto com dataclasses type-safe.

✅ Processamento em batch configurável para CPU e GPU.

✅ Exibição de vídeo configurável via parâmetro `show`.

✅ Qualidade facial configurável com 5 pesos independentes.
