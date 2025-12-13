"""
Interfaces do domínio (contratos para implementações de infraestrutura).
"""

from .face_detector import IFaceDetector
from .landmarks_detector import ILandmarksDetector

__all__ = ['IFaceDetector', 'ILandmarksDetector']
