# src/infrastructure/model/openvino_model_adapter.py
"""
Adaptador para modelos YOLO otimizados com OpenVINO.
"""

import logging
from typing import Iterator, Any
from pathlib import Path

from ultralytics import YOLO

from src.domain.services.model_interface import IDetectionModel


logger = logging.getLogger(__name__)


class OpenVINOModelAdapter(IDetectionModel):
    """
    Adaptador para modelos YOLO usando OpenVINO.
    Otimiza o modelo para inferência com FP16 ou outras precisões.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "AUTO",
        precision: str = "FP16"
    ):
        """
        Inicializa o adaptador OpenVINO.
        
        :param model_path: Caminho para o arquivo do modelo.
        :param device: Dispositivo OpenVINO (AUTO, CPU, GPU, NPU, etc).
        :param precision: Precisão do modelo (FP16, FP32, INT8).
        """
        self.model_path = model_path
        self.device = device
        self.precision = precision
        self._model = self._load_openvino_model()
        
        logger.info(
            f"Modelo OpenVINO carregado: {model_path} "
            f"(device={device}, precision={precision})"
        )
    
    def _load_openvino_model(self) -> YOLO:
        """
        Carrega e exporta o modelo para OpenVINO.
        
        O modelo OpenVINO é exportado uma vez e reutilizado. O dispositivo OpenVINO
        (AUTO, CPU, GPU, NPU) é selecionado automaticamente pelo runtime do OpenVINO
        durante a inferência, não durante o carregamento do modelo.
        
        :return: Modelo YOLO otimizado com OpenVINO.
        """
        model_path_obj = Path(self.model_path)
        
        # Determina o caminho do modelo OpenVINO exportado
        openvino_model_dir = model_path_obj.parent / f"{model_path_obj.stem}_openvino_model"
        openvino_model_path = openvino_model_dir / f"{model_path_obj.stem}.xml"
        
        # Se o modelo OpenVINO já existe, carrega diretamente
        if openvino_model_path.exists():
            logger.info(
                f"Modelo OpenVINO já exportado encontrado: {openvino_model_dir} "
                f"(device será selecionado automaticamente: {self.device})"
            )
            # IMPORTANTE: Ultralytics espera o caminho do DIRETÓRIO, não do arquivo .xml
            model = YOLO(str(openvino_model_dir), task="detect")
            logger.info(f"Modelo OpenVINO carregado com {len(model.names)} classes: {list(model.names.values())}")
        else:
            # Carrega o modelo original e exporta para OpenVINO
            logger.info(f"Carregando modelo original: {self.model_path}")
            original_model = YOLO(self.model_path)
            
            logger.info(
                f"Exportando modelo para OpenVINO "
                f"(precisão={self.precision}, device={self.device})..."
            )
            
            # export() retorna o caminho do DIRETÓRIO contendo os arquivos .xml e .bin
            export_path = original_model.export(
                format="openvino",
                half=(self.precision == "FP16"),  # FP16 precision
                int8=(self.precision == "INT8"),  # INT8 precision
                dynamic=False,  # Static shapes para melhor performance
                simplify=True
            )
            
            logger.info(f"Modelo exportado com sucesso para: {export_path}")
            
            # Recarrega o modelo exportado
            model = YOLO(export_path, task="detect")
            logger.info(f"Modelo OpenVINO carregado com {len(model.names)} classes: {list(model.names.values())}")
        
        return model
    
    def track(
        self,
        source: str,
        tracker: str,
        persist: bool = True,
        conf: float = 0.1,
        iou: float = 0.2,
        show: bool = False,
        stream: bool = True,
        batch: int = 4,
        verbose: bool = False,
        imgsz: int = 640
    ) -> Iterator[Any]:
        """
        Realiza tracking usando modelo OpenVINO otimizado.
        
        Nota: O dispositivo OpenVINO já foi selecionado durante o export do modelo.
        Não passamos 'device' aqui pois o modelo .xml já está compilado.
        
        IMPORTANTE: show e verbose são forçados para False para evitar erros de
        metadados ausentes no modelo exportado.
        """
        # Avisa se show ou verbose foram solicitados
        if show or verbose:
            logger.warning(
                "OpenVINO: parâmetros 'show' e 'verbose' foram desabilitados "
                "para evitar erros de metadados. Use PyTorch padrão se precisar dessas funcionalidades."
            )
        
        # Não passa 'device' - modelo OpenVINO já está compilado para o dispositivo
        return self._model.track(
            source=source,
            tracker=tracker,
            persist=persist,
            conf=conf,
            iou=iou,
            show=False,
            stream=stream,
            batch=batch,
            verbose=False,
            imgsz=imgsz
        )
    
    def get_model_info(self) -> dict:
        """
        Retorna informações sobre o modelo OpenVINO.
        """
        return {
            "type": "YOLO",
            "backend": "OpenVINO",
            "model_path": self.model_path,
            "device": self.device,
            "precision": self.precision,
            "optimization": "OpenVINO",
            "classes": len(self._model.names),
            "show_disabled": "Desabilitado para evitar erros de metadados"
        }