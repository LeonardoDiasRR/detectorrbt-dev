# src/infrastructure/model/__init__.py
"""
Módulo de infraestrutura para modelos de detecção.
"""

from src.infrastructure.model.model_factory import ModelFactory
from src.infrastructure.model.yolo_model_adapter import YOLOModelAdapter
from src.infrastructure.model.openvino_model_adapter import OpenVINOModelAdapter

__all__ = [
    "ModelFactory",
    "YOLOModelAdapter",
    "OpenVINOModelAdapter"
]