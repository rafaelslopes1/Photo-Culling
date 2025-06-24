#!/usr/bin/env python3
"""
Person Detection Module for Photo Culling System
Módulo de detecção de pessoas para sistema de seleção de fotos

Uses MediaPipe for person detection and pose analysis
Following the roadmap Phase 1 specifications
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple, NamedTuple
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


class PersonDetector:
    """
    Person detection and analysis using MediaPipe
    Implements dominant person identification and quality assessment
    Optimized for Mac M3 GPU acceleration when available
    """
    
    def __init__(self):
        """Initialize MediaPipe components with GPU optimization"""
        try:
            # Setup GPU optimization for Mac M3
            try:
                from ..utils.gpu_optimizer import MacM3Optimizer
                config, system_info = MacM3Optimizer.setup_quiet_and_optimized()
                
                if config['gpu_enabled']:
                    logger.info(f"GPU aceleração ativada - {system_info['chip_type']} com {system_info['gpu_cores']} cores")
                else:
                    logger.info("Usando otimização de CPU")
                    
            except ImportError:
                logger.info("Usando configuração padrão")
            
            # Initialize MediaPipe
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_hands = mp.solutions.hands
            self.mp_face_detection = mp.solutions.face_detection
            
            # Initialize pose detection for multiple persons
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.3  # Lower threshold to detect more people
            )
            
            # Initialize face detection with higher sensitivity
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,  # 1 for full-range detection (better for multiple people)
                min_detection_confidence=0.3  # Lower threshold for more detections
            )
            
            self.initialized = True
            logger.info("PersonDetector inicializado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar PersonDetector: {e}")
            self.initialized = False
    
    def detect_persons_and_faces(self, image: np.ndarray) -> Dict:
        """
        Detect multiple persons and faces using MediaPipe
        
        Args:
            image: Input image in BGR format (OpenCV format)
            
        Returns:
            Dictionary containing detection results with all persons
        """
        if not self.initialized:
            return self._get_empty_detection_result()
        
        try:
            if image is None:
                raise ValueError("Imagem não pode ser None")
            
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width = image.shape[:2]
            
            # Detect faces first (more reliable for multiple people)
            face_results = self.face_detection.process(rgb_image)
            
            # Process detections
            persons = []
            faces = []
            
            # Process face detections and estimate persons from faces
            if face_results.detections:
                for i, detection in enumerate(face_results.detections):
                    face_bbox = self._extract_face_bbox(detection, image_width, image_height)
                    if face_bbox:
                        # Store face detection
                        faces.append({
                            'id': i,
                            'bbox': face_bbox,
                            'confidence': detection.score[0],
                            'landmarks': self._extract_face_landmarks(detection, image_width, image_height)
                        })
                        
                        # Estimate person from face
                        person_bbox = self._estimate_person_from_face_bbox(
                            face_bbox, image_width, image_height
                        )
                        
                        if person_bbox:
                            # Try to get pose landmarks for this region if possible
                            person_landmarks = self._try_extract_pose_for_region(
                                rgb_image, person_bbox
                            )
                            
                            person = self._create_person_detection(
                                i, person_bbox, person_landmarks, 
                                image, image_width, image_height
                            )
                            persons.append(person)
            
            # Also try global pose detection for cases where faces aren't detected
            pose_results = self.pose.process(rgb_image)
            if pose_results.pose_landmarks and len(persons) == 0:
                # Only use global pose if no face-based persons were found
                person_bbox = self._extract_person_bbox_from_pose(
                    pose_results.pose_landmarks, image_width, image_height
                )
                if person_bbox:
                    person = self._create_person_detection(
                        0, person_bbox, pose_results.pose_landmarks, 
                        image, image_width, image_height
                    )
                    persons.append(person)
            
            # Identify dominant person
            dominant_person = None
            if persons:
                dominant_person = max(persons, key=lambda p: p.dominance_score)
            
            return {
                'persons': persons,
                'faces': faces,
                'dominant_person': dominant_person,
                'total_persons': len(persons),
                'total_faces': len(faces),
                'analysis_version': '2.0'  # Updated version for multi-person
            }
            
        except Exception as e:
            logger.error(f"Erro na detecção de pessoas: {e}")
            return self._get_empty_detection_result()
    
    def _extract_person_bbox_from_pose(self, pose_landmarks, 
                                     image_width: int, image_height: int) -> Optional[Tuple[int, int, int, int]]:
        """Extract bounding box from pose landmarks"""
        try:
            if not pose_landmarks:
                return None
            
            # Get all landmark coordinates
            x_coords = []
            y_coords = []
            
            for landmark in pose_landmarks.landmark:
                if landmark.visibility > 0.5:  # Only use visible landmarks
                    x_coords.append(landmark.x * image_width)
                    y_coords.append(landmark.y * image_height)
            
            if not x_coords:
                return None
            
            # Calculate bounding box
            x_min = max(0, int(min(x_coords)) - 20)  # Add some padding
            y_min = max(0, int(min(y_coords)) - 20)
            x_max = min(image_width, int(max(x_coords)) + 20)
            y_max = min(image_height, int(max(y_coords)) + 20)
            
            width = x_max - x_min
            height = y_max - y_min
            
            if width > 0 and height > 0:
                return (x_min, y_min, width, height)
            
            return None
            
        except Exception as e:
            logger.error(f"Erro ao extrair bbox da pose: {e}")
            return None
    
    def _extract_face_bbox(self, detection, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
        """Extract face bounding box from MediaPipe detection"""
        bbox = detection.location_data.relative_bounding_box
        x = int(bbox.xmin * image_width)
        y = int(bbox.ymin * image_height)
        w = int(bbox.width * image_width)
        h = int(bbox.height * image_height)
        return (x, y, w, h)
    
    def _extract_face_landmarks(self, detection, image_width: int, image_height: int) -> List[Tuple[float, float]]:
        """Extract face landmarks from MediaPipe detection"""
        landmarks = []
        for landmark in detection.location_data.relative_keypoints:
            x = landmark.x * image_width
            y = landmark.y * image_height
            landmarks.append((x, y))
        return landmarks
    
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
            centrality = 1 - (distance / max_distance)
            
            # Calculate local sharpness (simplified variance of Laplacian)
            roi = image[y:y+h, x:x+w]
            local_sharpness = 0.0
            if roi.size > 0:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                local_sharpness = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
                # Normalize to 0-1 range (typical blur scores are 0-2000)
                local_sharpness = min(1.0, local_sharpness / 2000.0)
            
            # Extract landmarks
            landmarks = []
            if pose_landmarks:
                if hasattr(pose_landmarks, 'landmark'):  # MediaPipe format
                    for landmark in pose_landmarks.landmark:
                        landmarks.append((landmark.x * image_width, landmark.y * image_height))
                elif isinstance(pose_landmarks, list):  # Already converted format
                    landmarks = pose_landmarks
            
            # Calculate dominance score
            dominance_score = self._calculate_person_dominance(
                bbox, (image_height, image_width), local_sharpness
            )
            
            return PersonDetection(
                person_id=person_id,
                dominance_score=dominance_score,
                bounding_box=bbox,
                landmarks=landmarks,
                pose_landmarks=landmarks,
                confidence=0.8,  # Default confidence for pose detection
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
        """
        Calculate dominance score for a detected person
        Based on: area_ratio × 0.4 + centrality × 0.3 + local_sharpness × 0.3
        """
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
    
    def identify_dominant_person(self, persons_data: List[PersonDetection]) -> Optional[PersonDetection]:
        """
        Identify the dominant person in the image based on multiple criteria
        """
        if not persons_data:
            return None
        
        # Return person with highest dominance score
        return max(persons_data, key=lambda p: p.dominance_score)
    
    def _get_empty_detection_result(self) -> Dict:
        """Return empty result in case of detection failure"""
        return {
            'persons': [],
            'faces': [],
            'dominant_person': None,
            'total_persons': 0,
            'total_faces': 0,
            'analysis_version': '1.0',
            'error': True
        }
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        try:
            if hasattr(self, 'pose'):
                self.pose.close()
            if hasattr(self, 'face_detection'):
                self.face_detection.close()
        except:
            pass

    def _estimate_person_from_face_bbox(self, face_bbox: Tuple[int, int, int, int], 
                                       image_width: int, image_height: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Estimate person bounding box from face bounding box
        
        Args:
            face_bbox: Face bounding box (x, y, w, h)
            image_width: Image width
            image_height: Image height
            
        Returns:
            Estimated person bounding box (x, y, w, h) or None
        """
        try:
            face_x, face_y, face_w, face_h = face_bbox
            
            # Calculate face center
            face_center_x = face_x + face_w // 2
            face_center_y = face_y + face_h // 2
            
            # Estimate person dimensions based on typical proportions
            # Face is roughly 1/8 of total body height
            estimated_body_height = face_h * 8
            estimated_body_width = face_w * 2.5  # Rough body width estimate
            
            # Position person bbox (center it on face center)
            person_x = max(0, int(face_center_x - estimated_body_width // 2))
            person_y = max(0, int(face_y - face_h * 0.5))  # Start slightly above face
            person_w = min(int(estimated_body_width), image_width - person_x)
            person_h = min(int(estimated_body_height), image_height - person_y)
            
            # Ensure minimum size
            if person_w < 50 or person_h < 100:
                return None
            
            return (person_x, person_y, person_w, person_h)
            
        except Exception as e:
            logger.error(f"Erro ao estimar pessoa a partir da face: {e}")
            return None
    
    def _try_extract_pose_for_region(self, rgb_image: np.ndarray, 
                                   person_bbox: Tuple[int, int, int, int]) -> Optional[List]:
        """
        Try to extract pose landmarks for a specific region
        
        Args:
            rgb_image: RGB image
            person_bbox: Person bounding box to analyze
            
        Returns:
            Pose landmarks if found, None otherwise
        """
        try:
            x, y, w, h = person_bbox
            
            # Extract region of interest
            roi = rgb_image[y:y+h, x:x+w]
            
            if roi.size == 0:
                return None
            
            # Try pose detection on the region
            pose_results = self.pose.process(roi)
            
            if pose_results.pose_landmarks:
                # Convert landmarks back to full image coordinates
                adjusted_landmarks = []
                for landmark in pose_results.pose_landmarks.landmark:
                    # Convert normalized coordinates to full image coordinates
                    full_x = x + landmark.x * w
                    full_y = y + landmark.y * h
                    adjusted_landmarks.append((full_x, full_y))
                
                return adjusted_landmarks
            
            return None
            
        except Exception as e:
            logger.debug(f"Não foi possível extrair pose para região: {e}")
            return None


def detect_persons_in_image(image_path: str) -> Optional[Dict]:
    """
    Convenience function to detect persons in an image file
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Person detection results or None if failed
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Não foi possível carregar a imagem: {image_path}")
            return None
        
        detector = PersonDetector()
        return detector.detect_persons_and_faces(image)
        
    except Exception as e:
        logger.error(f"Erro ao detectar pessoas na imagem {image_path}: {e}")
        return None


if __name__ == "__main__":
    # Test the module
    import sys
    if len(sys.argv) > 1:
        result = detect_persons_in_image(sys.argv[1])
        if result:
            print("Person Detection Results:")
            print(f"  Total persons: {result['total_persons']}")
            print(f"  Total faces: {result['total_faces']}")
            if result['dominant_person']:
                dp = result['dominant_person']
                print(f"  Dominant person dominance score: {dp.dominance_score:.3f}")
                print(f"  Dominant person bbox: {dp.bounding_box}")
        else:
            print("Failed to detect persons")
