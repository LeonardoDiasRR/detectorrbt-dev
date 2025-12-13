"""
Implementações de Machine Learning (Infrastructure Layer).
"""

from .yolo_face_detector import YOLOFaceDetector
from .yolo_landmarks_detector import YOLOLandmarksDetector

__all__ = ['YOLOFaceDetector', 'YOLOLandmarksDetector']
