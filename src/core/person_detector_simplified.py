#!/usr/bin/env python3
"""
Simplified Person Detector without MediaPipe
Detector de pessoas simplificado sem MediaPipe

Uses OpenCV face detection as fallback when MediaPipe is not available
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PersonDetection:
    """Data class for person detection results"""
    person_id: int
    dominance_score: float
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    landmarks: List[Tuple[float, float]]
    pose_landmarks: Optional[List[Tuple[float, float]]] = None
    confidence: float = 0.0
    area_ratio: float = 0.0
    centrality: float = 0.0
    local_sharpness: float = 0.0
    
    def __post_init__(self):
        """Convert numpy types to native Python types for JSON serialization"""
        self.person_id = int(self.person_id)
        self.dominance_score = float(self.dominance_score)
        # Ensure bounding_box has exactly 4 elements
        bbox = list(self.bounding_box)
        if len(bbox) == 4:
            self.bounding_box = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        self.landmarks = [(float(x), float(y)) for x, y in self.landmarks]
        if self.pose_landmarks:
            self.pose_landmarks = [(float(x), float(y)) for x, y in self.pose_landmarks]
        self.confidence = float(self.confidence)
        self.area_ratio = float(self.area_ratio)
        self.centrality = float(self.centrality)
        self.local_sharpness = float(self.local_sharpness)


class SimplifiedPersonDetector:
    """
    Simplified person detection using only OpenCV
    Fallback when MediaPipe is not available
    """
    
    def __init__(self):
        """Initialize OpenCV face detector"""
        try:
            # Try different paths for face cascade
            cascade_paths = [
                'haarcascade_frontalface_default.xml',
                '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                '/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
            ]
            
            self.face_cascade = None
            for path in cascade_paths:
                try:
                    cascade = cv2.CascadeClassifier(path)
                    if not cascade.empty():
                        self.face_cascade = cascade
                        break
                except:
                    continue
            
            self.initialized = self.face_cascade is not None and not self.face_cascade.empty()
            if self.initialized:
                logger.info("SimplifiedPersonDetector inicializado com sucesso (OpenCV)")
            else:
                logger.warning("Não foi possível carregar o detector de faces")
                
        except Exception as e:
            logger.error(f"Erro ao inicializar SimplifiedPersonDetector: {e}")
            self.initialized = False
            self.face_cascade = None
    
    def detect_persons_and_faces(self, image: np.ndarray) -> Dict:
        """
        Detect faces using OpenCV and estimate persons
        
        Args:
            image: Input image in BGR format (OpenCV format)
            
        Returns:
            Dictionary containing detection results
        """
        if not self.initialized:
            return self._get_empty_detection_result()
        
        try:
            if image is None:
                raise ValueError("Imagem não pode ser None")
            
            if not self.initialized or self.face_cascade is None:
                logging.warning("Face cascade not initialized")
                return self._get_empty_detection_result()
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_height, image_width = image.shape[:2]
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            # Convert faces to persons (estimate body from face)
            persons = []
            face_detections = []
            
            for i, (x, y, w, h) in enumerate(faces):
                # Create face detection
                face_detections.append({
                    'id': i,
                    'bbox': (x, y, w, h),
                    'confidence': 0.8,  # Default confidence for Haar cascades
                    'landmarks': [(x + w//2, y + h//2)]  # Center point as landmark
                })
                
                # Estimate person bbox from face (rough estimation)
                person_bbox = self._estimate_person_from_face((x, y, w, h), image_width, image_height)
                
                if person_bbox:
                    person = self._create_person_detection(
                        i, person_bbox, [], image, image_width, image_height
                    )
                    persons.append(person)
            
            # Identify dominant person
            dominant_person = None
            if persons:
                dominant_person = max(persons, key=lambda p: p.dominance_score)
            
            return {
                'persons': persons,
                'faces': face_detections,
                'dominant_person': dominant_person,
                'total_persons': len(persons),
                'total_faces': len(faces),
                'analysis_version': '1.0_simplified'
            }
            
        except Exception as e:
            logger.error(f"Erro na detecção de pessoas (simplified): {e}")
            return self._get_empty_detection_result()
    
    def _estimate_person_from_face(self, face_bbox: Tuple[int, int, int, int], 
                                 image_width: int, image_height: int) -> Optional[Tuple[int, int, int, int]]:
        """Estimate person bounding box from face detection"""
        try:
            fx, fy, fw, fh = face_bbox
            
            # Rough estimation: person is typically 6-8 times the face height
            # and face is typically in upper 1/4 of person
            person_height = int(fh * 7)  # 7x face height
            person_width = int(fw * 2.5)  # 2.5x face width
            
            # Position person bbox
            person_x = max(0, fx - (person_width - fw) // 2)
            person_y = max(0, fy - fh // 4)  # Face is 1/4 from top
            
            # Ensure bbox is within image bounds
            person_x = min(person_x, image_width - person_width)
            person_y = min(person_y, image_height - person_height)
            person_width = min(person_width, image_width - person_x)
            person_height = min(person_height, image_height - person_y)
            
            if person_width > 0 and person_height > 0:
                return (person_x, person_y, person_width, person_height)
            
            return None
            
        except Exception as e:
            logger.error(f"Erro ao estimar pessoa da face: {e}")
            return None
    
    def _create_person_detection(self, person_id: int, bbox: Tuple[int, int, int, int],
                               pose_landmarks, image: np.ndarray, 
                               image_width: int, image_height: int) -> PersonDetection:
        """Create PersonDetection object with calculated metrics"""
        try:
            x, y, w, h = bbox
            
            # Calculate area ratio
            person_area = w * h
            image_area = image_width * image_height
            area_ratio = person_area / image_area
            
            # Calculate centrality (distance from center)
            center_x, center_y = image_width // 2, image_height // 2
            person_center_x, person_center_y = x + w//2, y + h//2
            max_distance = np.sqrt(center_x**2 + center_y**2)
            distance = np.sqrt((person_center_x - center_x)**2 + (person_center_y - center_y)**2)
            centrality = 1 - (distance / max_distance) if max_distance > 0 else 0
            
            # Calculate local sharpness (simplified variance of Laplacian)
            roi = image[y:y+h, x:x+w]
            local_sharpness = 0.0
            if roi.size > 0:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                local_sharpness = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
                # Normalize to 0-1 range (typical blur scores are 0-2000)
                local_sharpness = min(1.0, local_sharpness / 2000.0)
            
            # Calculate dominance score
            dominance_score = self._calculate_person_dominance(
                bbox, (image_height, image_width), local_sharpness
            )
            
            return PersonDetection(
                person_id=person_id,
                dominance_score=dominance_score,
                bounding_box=bbox,
                landmarks=[(x + w//2, y + h//2)],  # Center point
                pose_landmarks=None,
                confidence=0.7,  # Default confidence for estimated person
                area_ratio=area_ratio,
                centrality=centrality,
                local_sharpness=local_sharpness
            )
            
        except Exception as e:
            logger.error(f"Erro ao criar PersonDetection: {e}")
            return PersonDetection(
                person_id=person_id,
                dominance_score=0.0,
                bounding_box=bbox,
                landmarks=[],
                confidence=0.0
            )
    
    def _calculate_person_dominance(self, person_bbox: Tuple[int, int, int, int], 
                                  image_shape: Tuple[int, int], 
                                  local_sharpness: float) -> float:
        """Calculate dominance score for a detected person"""
        try:
            x, y, w, h = person_bbox
            img_h, img_w = image_shape
            
            # Area ratio (how much of the image the person occupies)
            area_ratio = (w * h) / (img_w * img_h)
            
            # Centrality (how close to center the person is)
            center_x, center_y = img_w // 2, img_h // 2
            person_center_x, person_center_y = x + w//2, y + h//2
            max_distance = np.sqrt(center_x**2 + center_y**2)
            distance = np.sqrt((person_center_x - center_x)**2 + (person_center_y - center_y)**2)
            centrality = 1 - (distance / max_distance) if max_distance > 0 else 0
            
            # Combined dominance score (following roadmap formula)
            dominance_score = (area_ratio * 0.4 + centrality * 0.3 + local_sharpness * 0.3)
            
            return min(1.0, max(0.0, dominance_score))  # Clamp to [0, 1]
            
        except Exception as e:
            logger.error(f"Erro ao calcular dominância: {e}")
            return 0.0
    
    def _get_empty_detection_result(self) -> Dict:
        """Return empty result in case of detection failure"""
        return {
            'persons': [],
            'faces': [],
            'dominant_person': None,
            'total_persons': 0,
            'total_faces': 0,
            'analysis_version': '1.0_simplified',
            'error': True
        }


# Factory function to create appropriate detector
def create_person_detector():
    """Create the best available person detector"""
    try:
        # Try MediaPipe first
        import mediapipe as mp
        from .person_detector import PersonDetector
        return PersonDetector()
    except ImportError:
        # Fallback to simplified detector
        logger.info("MediaPipe não disponível, usando detector simplificado")
        return SimplifiedPersonDetector()


def detect_persons_in_image_simplified(image_path: str) -> Optional[Dict]:
    """
    Convenience function to detect persons in an image file using simplified detector
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Não foi possível carregar a imagem: {image_path}")
            return None
        
        detector = SimplifiedPersonDetector()
        return detector.detect_persons_and_faces(image)
        
    except Exception as e:
        logger.error(f"Erro ao detectar pessoas na imagem {image_path}: {e}")
        return None
