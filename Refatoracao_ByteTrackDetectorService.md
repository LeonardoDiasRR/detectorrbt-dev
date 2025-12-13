# PROPOSTA DE ESTRUTURA OTIMIZADA PARA DDD do bytetrack_detector_service

# ESTRUTURA DE DIRETÓRIOS
src/
├── application/
│   └── use_cases/
│       ├── load_cameras_use_case.py              # ✅ Já existe
│       └── process_face_detection_use_case.py    # ← NOVO (substitui bytetrack_detector_service)
│
├── domain/
│   ├── entities/
│   │   ├── camera_entity.py                      # ✅ Já existe
│   │   ├── track_entity.py                       # ✅ Já existe
│   │   ├── event_entity.py                       # ✅ Já existe
│   │   └── frame_entity.py                       # ✅ Já existe
│   │
│   ├── value_objects/
│   │   ├── bbox_vo.py                            # ✅ Já existe
│   │   ├── confidence_vo.py                      # ✅ Já existe
│   │   └── landmarks_vo.py                       # ✅ Já existe
│   │
│   ├── services/                                 # Regras de negócio
│   │   ├── track_lifecycle_service.py            # ← NOVO: Gerencia ciclo de vida do track
│   │   ├── event_creation_service.py             # ← NOVO: Cria eventos a partir de detecções
│   │   ├── track_validation_service.py           # ← NOVO: Valida tracks (movimento, confiança)
│   │   ├── face_quality_service.py               # ✅ Já existe
│   │   └── movement_detection_service.py         # ← NOVO: Detecta movimento significativo
│   │
│   ├── interfaces/                               # ← NOVA pasta
│   │   ├── face_detector.py                      # Interface para detectores
│   │   ├── landmarks_detector.py                 # Interface para landmarks
│   │   └── video_reader.py                       # Interface para leitura de vídeo
│   │
│   ├── repositories/                             # ✅ Já existe
│   │   └── camera_repository.py
│   │
│   └── adapters/                                 # ✅ Já existe
│       └── findface_adapter.py
│
└── infrastructure/
    ├── ml/                                       # ← NOVA pasta
    │   ├── yolo_face_detector.py                 # Implementa IFaceDetector
    │   └── yolo_landmarks_detector.py            # Implementa ILandmarksDetector
    │
    ├── video/                                    # ← NOVA pasta
    │   └── rtsp_stream_reader.py                 # Implementa IVideoReader
    │
    ├── storage/                                  # ← NOVA pasta
    │   └── async_image_writer.py                 # ImageSaveService movido aqui
    │
    ├── model/                                    # ✅ Já existe
    │   ├── model_factory.py
    │   └── landmarks_model_factory.py
    │
    ├── repositories/                             # ✅ Já existe
    │   └── camera_repository_findface.py
    │
    └── clients/                                  # ✅ Já existe
        └── findface_multi.py
		
		
# FLUXO DE RESPONSABILIDADES
## 1. ProcessFaceDetectionUseCase (Application)
- Orquestra o fluxo completo
- Injeta dependências (detector, video reader, services)
- Gerencia threads e workers

## 2. TrackLifecycleService (Domain)
- Adiciona evento ao track
- Decide quando finalizar track
- Obtém melhor evento

## 3. EventCreationService (Domain)
- Cria Event a partir de detecção YOLO
- Calcula face quality (usa FaceQualityService)
- Extrai landmarks

## 4. TrackValidationService (Domain)
- Valida movimento (usa MovementDetectionService)
- Valida confiança mínima
- Valida bbox mínimo

## 5. YOLOFaceDetector (Infrastructure)
- Implementa IFaceDetector
- Wrapper para YOLO .track()
- Retorna detecções brutas

## 6. RTSPStreamReader (Infrastructure)
- Implementa IVideoReader
- Lê frames do RTSP
- Gerencia reconexão


# ETAPAS

## Etapa 1 - Criar as interfaces no domínio
- ✅ Criar src/domain/interfaces/face_detector.py com IFaceDetector abstrata
- ✅ Criar src/domain/interfaces/landmarks_detector.py com ILandmarksDetector
- ✅ Criar src/domain/interfaces/video_reader.py com IVideoReader
- ✅ Benefício: Domain não depende mais de YOLO diretamente (Inversão de Dependência)

## Etapa 2 - Implementar interfaces para YOLO e RTSP
- ✅ Criar src/infrastructure/ml/yolo_face_detector.py implementando IFaceDetector
- ✅ Criar src/infrastructure/ml/yolo_landmarks_detector.py implementando ILandmarksDetector
- ✅ Criar src/infrastructure/video/rtsp_stream_reader.py implementando IVideoReader
- ✅ Benefício: Infraestrutura isolada, fácil trocar YOLO por outro detector

## Etapa 3 - Extrair lógica para domain services aos poucos
- ✅ Extrair validação de tracks → TrackValidationService
- ✅ Extrair criação de eventos → EventCreationService
- ✅ Extrair detecção de movimento → MovementDetectionService
- ✅ Extrair ciclo de vida do track → TrackLifecycleService
- ✅ Benefício: Regras de negócio isoladas, testáveis, reutilizáveis

## Etapa 4 - Separar responsabilidades em services menores
Quebrar ByteTrackDetectorService (1500+ linhas) em:
- ✅ EventCreationService - cria eventos de detecções (CRIADO e INTEGRADO no Use Case)
- ✅ TrackLifecycleService - gerencia tracks (add, finalize) (CRIADO e INTEGRADO no Use Case)
- ✅ TrackValidationService - valida tracks antes de enviar (CRIADO e INTEGRADO no Use Case)
- ✅ MovementDetectionService - detecta movimento significativo (CRIADO e INTEGRADO no Use Case)
- ✅ Benefício: Single Responsibility Principle, cada classe faz UMA coisa

## Etapa 5 - Migrar ByteTrackDetectorService → ProcessFaceDetectionUseCase
- ✅ Criar src/application/use_cases/process_face_detection_use_case.py
- ✅ Injetar dependencies: IFaceDetector, IVideoReader, domain services
- ✅ Orquestrar fluxo: lê frame → detecta → cria evento → valida → envia FindFace
- ✅ Migrar run.py para usar novo use case (CONCLUÍDO)
- ⏳ Manter ByteTrackDetectorService como wrapper temporário (retrocompatibilidade)
- ✅ Benefício: Camada de aplicação separada do domínio, orquestração clara

**NOTA:** ProcessFaceDetectionUseCase criado e funcionando. run.py atualizado para usar nova arquitetura DDD.


Antes:
├── ByteTrackDetectorService (1500 linhas, faz TUDO)

Depois:
├── ProcessFaceDetectionUseCase (orquestra)
│   ├── IFaceDetector (abstração)
│   ├── IVideoReader (abstração)
│   ├── EventCreationService (regra de negócio)
│   ├── TrackLifecycleService (regra de negócio)
│   ├── TrackValidationService (regra de negócio)
│   └── MovementDetectionService (regra de negócio)


---

## MIGRAÇÃO COMPLETA - run.py Atualizado

### Mudanças Implementadas no run.py:

1. **Imports Atualizados:**
   - ✅ Removido: `ByteTrackDetectorService`
   - ✅ Adicionado: `ProcessFaceDetectionUseCase`
   - ✅ Adicionado: Domain services (TrackValidationService, EventCreationService, MovementDetectionService, TrackLifecycleService)
   - ✅ Adicionado: Wrappers de infraestrutura (YOLOFaceDetector, YOLOLandmarksDetector, RTSPStreamReader)

2. **Instanciação de Serviços de Domínio (Compartilhados):**
   ```python
   track_validation_service = TrackValidationService(...)
   event_creation_service = EventCreationService()
   movement_detection_service = MovementDetectionService()
   track_lifecycle_service = TrackLifecycleService(...)
   face_quality_service = FaceQualityService()
   ```

3. **Loop de Criação de Processadores (Por Câmera):**
   - Carrega modelo YOLO → Cria `YOLOFaceDetector` wrapper
   - Carrega modelo de landmarks → Cria `YOLOLandmarksDetector` wrapper
   - Cria `RTSPStreamReader` com URL da câmera
   - Instancia `ProcessFaceDetectionUseCase` injetando todas as dependências

4. **Injeção de Dependências no Use Case:**
   - `face_detector`: IFaceDetector (YOLOFaceDetector)
   - `video_reader`: IVideoReader (RTSPStreamReader)
   - `findface_adapter`: FindfaceAdapter
   - `findface_queue`: Fila global compartilhada
   - `landmarks_detector`: ILandmarksDetector (YOLOLandmarksDetector ou None)
   - `image_save_service`: ImageSaveService compartilhado
   - `face_quality_service`: FaceQualityService compartilhado
   - Domain services compartilhados (não instanciados por câmera)

### Benefícios da Migração:

✅ **Separação de Camadas:** Domain não depende de YOLO/RTSP (Dependency Inversion)  
✅ **Testabilidade:** Services isolados podem ser testados unitariamente  
✅ **Manutenibilidade:** Cada service tem responsabilidade única (SRP)  
✅ **Flexibilidade:** Fácil trocar YOLO por outro detector (basta implementar IFaceDetector)  
✅ **Reutilização:** Services de domínio compartilhados entre câmeras  

### Estado Atual:

- ✅ **ProcessFaceDetectionUseCase:** Implementado e integrado
- ✅ **run.py:** Migrado para nova arquitetura DDD
- ⚠️ **ByteTrackDetectorService:** Mantido no código para retrocompatibilidade (pode ser removido futuramente)
- ✅ **Arquitetura DDD:** Completa e funcional

### Próximos Passos Opcionais:

1. **Testes Unitários:** Criar testes para cada domain service
2. **Remover ByteTrackDetectorService:** Após validação completa da nova arquitetura
3. **Documentação:** Adicionar docstrings detalhadas em todos os services
4. **Métricas:** Implementar logging de performance para comparar com versão anterior
