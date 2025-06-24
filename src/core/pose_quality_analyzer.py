#!/usr/bin/env python3
"""
Pose Quality Analyzer Module for Photo Culling System
Módulo de análise de qualidade de pose para sistema de seleção de fotos

Analyzes pose quality, naturalness, and facial orientation for people in images
Following the Phase 2 roadmap specifications
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)


class PoseNaturalness(Enum):
    """Naturalness levels for pose analysis"""
    VERY_NATURAL = "very_natural"
    NATURAL = "natural"
    SOMEWHAT_NATURAL = "somewhat_natural"
    FORCED = "forced"
    VERY_FORCED = "very_forced"


class FacialOrientation(Enum):
    """Facial orientation types"""
    FRONTAL = "frontal"
    THREE_QUARTER = "three_quarter"
    PROFILE = "profile"
    TILTED = "tilted"
    UNKNOWN = "unknown"


class MotionType(Enum):
    """Types of motion detected"""
    STATIC = "static"
    SLIGHT_MOVEMENT = "slight_movement"
    ACTIVE_MOVEMENT = "active_movement"
    MOTION_BLUR = "motion_blur"


@dataclass
class PoseQualityMetrics:
    """Data class for pose quality analysis results"""
    posture_quality_score: float
    facial_orientation: FacialOrientation
    naturalness_level: PoseNaturalness
    motion_type: MotionType
    pose_stability_score: float
    symmetry_score: float
    overall_pose_rating: str
    detailed_analysis: Dict


class PoseQualityAnalyzer:
    """
    Analyzer for pose quality, naturalness, and facial orientation
    Uses MediaPipe pose landmarks and facial landmarks for analysis
    """
    
    def __init__(self):
        """Initialize the pose quality analyzer"""
        # Thresholds for various pose quality metrics
        self.symmetry_threshold = 0.1  # Maximum asymmetry allowed
        self.naturalness_threshold = 0.6  # Minimum score for natural pose
        self.stability_threshold = 0.7  # Minimum score for stable pose
        
        # Weights for combining pose quality metrics
        self.pose_weights = {
            'posture': 0.3,      # Body posture quality
            'facial': 0.25,      # Facial orientation and expression
            'naturalness': 0.25, # How natural the pose looks
            'symmetry': 0.1,     # Body symmetry
            'stability': 0.1     # Pose stability/motion
        }
    
    def analyze_pose_quality(self, pose_landmarks: Optional[List], 
                           face_landmarks: Optional[List] = None,
                           image_shape: Optional[Tuple[int, int]] = None) -> PoseQualityMetrics:
        """
        Comprehensive pose quality analysis
        
        Args:
            pose_landmarks: MediaPipe pose landmarks (33 points)
            face_landmarks: MediaPipe face landmarks (468 points)
            image_shape: Shape of the image (height, width) for normalization
            
        Returns:
            PoseQualityMetrics with detailed analysis
        """
        try:
            if not pose_landmarks:
                logger.warning("Landmarks de pose não fornecidos")
                return self._get_default_pose_metrics()
            
            # Convert landmarks to numpy arrays for easier processing
            pose_points = self._extract_pose_points(pose_landmarks, image_shape)
            face_points = self._extract_face_points(face_landmarks, image_shape) if face_landmarks else None
            
            # Analyze body posture
            posture_score = self._analyze_body_posture(pose_points)
            
            # Analyze facial orientation
            facial_orientation = self._analyze_facial_orientation(face_points, pose_points)
            
            # Assess pose naturalness
            naturalness_level, naturalness_score = self._assess_pose_naturalness(pose_points)
            
            # Calculate symmetry
            symmetry_score = self._calculate_body_symmetry(pose_points)
            
            # Detect motion and stability
            motion_type, stability_score = self._analyze_motion_and_stability(pose_points)
            
            # Calculate overall pose rating
            overall_rating = self._calculate_overall_pose_rating(
                posture_score, naturalness_score, symmetry_score, stability_score
            )
            
            # Detailed analysis for debugging and insights
            detailed_analysis = {
                'posture_score': posture_score,
                'naturalness_score': naturalness_score,
                'joint_angles': self._calculate_joint_angles(pose_points),
                'body_alignment': self._analyze_body_alignment(pose_points),
                'gesture_analysis': self._analyze_gestures(pose_points)
            }
            
            return PoseQualityMetrics(
                posture_quality_score=posture_score,
                facial_orientation=facial_orientation,
                naturalness_level=naturalness_level,
                motion_type=motion_type,
                pose_stability_score=stability_score,
                symmetry_score=symmetry_score,
                overall_pose_rating=overall_rating,
                detailed_analysis=detailed_analysis
            )
            
        except Exception as e:
            logger.error(f"Erro na análise de qualidade de pose: {e}")
            return self._get_default_pose_metrics()
    
    def _extract_pose_points(self, landmarks: List, image_shape: Optional[Tuple[int, int]]) -> np.ndarray:
        """
        Extract pose landmarks as normalized numpy array
        """
        if not landmarks:
            return np.array([])
        
        # Extract x, y coordinates (ignore z for 2D analysis)
        points = []
        for landmark in landmarks:
            if hasattr(landmark, 'x') and hasattr(landmark, 'y'):
                points.append([landmark.x, landmark.y])
            else:
                # Assume it's already a coordinate pair
                points.append([landmark[0], landmark[1]])
        
        points = np.array(points)
        
        # If image shape is provided, convert to pixel coordinates
        if image_shape:
            height, width = image_shape
            points[:, 0] *= width
            points[:, 1] *= height
        
        return points
    
    def _extract_face_points(self, landmarks: List, image_shape: Optional[Tuple[int, int]]) -> Optional[np.ndarray]:
        """
        Extract face landmarks as normalized numpy array
        """
        if not landmarks:
            return None
        
        return self._extract_pose_points(landmarks, image_shape)
    
    def _analyze_body_posture(self, pose_points: np.ndarray) -> float:
        """
        Analyze overall body posture quality
        """
        try:
            if len(pose_points) < 11:  # Need at least torso landmarks
                return 0.5
            
            # MediaPipe pose landmark indices
            # 0: nose, 11: left_shoulder, 12: right_shoulder, 23: left_hip, 24: right_hip
            nose_idx, left_shoulder_idx, right_shoulder_idx = 0, 11, 12
            left_hip_idx, right_hip_idx = 23, 24
            
            if len(pose_points) <= max(nose_idx, left_shoulder_idx, right_shoulder_idx, left_hip_idx, right_hip_idx):
                return 0.5
            
            # Calculate spine alignment
            spine_score = self._calculate_spine_alignment(pose_points)
            
            # Calculate shoulder levelness
            shoulder_score = self._calculate_shoulder_levelness(pose_points)
            
            # Calculate hip alignment
            hip_score = self._calculate_hip_alignment(pose_points)
            
            # Check for natural arm positioning
            arm_score = self._analyze_arm_positioning(pose_points)
            
            # Combine posture scores
            posture_score = (spine_score * 0.3 + shoulder_score * 0.25 + 
                           hip_score * 0.25 + arm_score * 0.2)
            
            return max(0.0, min(1.0, posture_score))
            
        except Exception as e:
            logger.error(f"Erro ao analisar postura corporal: {e}")
            return 0.5
    
    def _calculate_spine_alignment(self, pose_points: np.ndarray) -> float:
        """
        Calculate spine alignment score based on shoulder-hip alignment
        """
        try:
            # Get shoulder and hip midpoints
            left_shoulder, right_shoulder = pose_points[11], pose_points[12]
            left_hip, right_hip = pose_points[23], pose_points[24]
            
            shoulder_midpoint = (left_shoulder + right_shoulder) / 2
            hip_midpoint = (left_hip + right_hip) / 2
            
            # Calculate spine vector
            spine_vector = shoulder_midpoint - hip_midpoint
            
            # Calculate angle from vertical (0 degrees = perfect vertical)
            angle_from_vertical = abs(math.atan2(spine_vector[0], spine_vector[1]))
            angle_degrees = math.degrees(angle_from_vertical)
            
            # Score based on how close to vertical (0-15 degrees = good)
            if angle_degrees <= 15:
                return 1.0
            elif angle_degrees <= 30:
                return 1.0 - (angle_degrees - 15) / 15 * 0.5
            else:
                return max(0.0, 0.5 - (angle_degrees - 30) / 60 * 0.5)
            
        except Exception as e:
            logger.error(f"Erro ao calcular alinhamento da coluna: {e}")
            return 0.5
    
    def _calculate_shoulder_levelness(self, pose_points: np.ndarray) -> float:
        """
        Calculate how level the shoulders are
        """
        try:
            left_shoulder, right_shoulder = pose_points[11], pose_points[12]
            
            # Calculate height difference
            height_diff = abs(left_shoulder[1] - right_shoulder[1])
            
            # Calculate shoulder width for normalization
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
            
            if shoulder_width == 0:
                return 0.5
            
            # Normalize height difference by shoulder width
            normalized_diff = height_diff / shoulder_width
            
            # Score based on levelness (0-0.1 = perfect, 0.1-0.2 = good, >0.2 = poor)
            if normalized_diff <= 0.1:
                return 1.0
            elif normalized_diff <= 0.2:
                return 1.0 - (normalized_diff - 0.1) / 0.1 * 0.5
            else:
                return max(0.0, 0.5 - (normalized_diff - 0.2) / 0.3 * 0.5)
            
        except Exception as e:
            logger.error(f"Erro ao calcular nivelamento dos ombros: {e}")
            return 0.5
    
    def _calculate_hip_alignment(self, pose_points: np.ndarray) -> float:
        """
        Calculate hip alignment score
        """
        try:
            left_hip, right_hip = pose_points[23], pose_points[24]
            
            # Similar to shoulder levelness
            height_diff = abs(left_hip[1] - right_hip[1])
            hip_width = abs(left_hip[0] - right_hip[0])
            
            if hip_width == 0:
                return 0.5
            
            normalized_diff = height_diff / hip_width
            
            if normalized_diff <= 0.1:
                return 1.0
            elif normalized_diff <= 0.2:
                return 1.0 - (normalized_diff - 0.1) / 0.1 * 0.5
            else:
                return max(0.0, 0.5 - (normalized_diff - 0.2) / 0.3 * 0.5)
            
        except Exception as e:
            logger.error(f"Erro ao calcular alinhamento do quadril: {e}")
            return 0.5
    
    def _analyze_arm_positioning(self, pose_points: np.ndarray) -> float:
        """
        Analyze arm positioning for naturalness
        """
        try:
            # MediaPipe arm landmarks: 11,12 (shoulders), 13,14 (elbows), 15,16 (wrists)
            left_shoulder, right_shoulder = pose_points[11], pose_points[12]
            left_elbow, right_elbow = pose_points[13], pose_points[14]
            left_wrist, right_wrist = pose_points[15], pose_points[16]
            
            # Calculate arm angles
            left_arm_score = self._calculate_arm_naturalness(left_shoulder, left_elbow, left_wrist)
            right_arm_score = self._calculate_arm_naturalness(right_shoulder, right_elbow, right_wrist)
            
            # Average both arms
            return (left_arm_score + right_arm_score) / 2
            
        except Exception as e:
            logger.error(f"Erro ao analisar posicionamento dos braços: {e}")
            return 0.5
    
    def _calculate_arm_naturalness(self, shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float:
        """
        Calculate naturalness score for a single arm
        """
        try:
            # Calculate upper arm and forearm vectors
            upper_arm = elbow - shoulder
            forearm = wrist - elbow
            
            # Calculate elbow angle
            dot_product = np.dot(upper_arm, forearm)
            norms = np.linalg.norm(upper_arm) * np.linalg.norm(forearm)
            
            if norms == 0:
                return 0.5
            
            cos_angle = dot_product / norms
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = math.acos(cos_angle)
            angle_degrees = math.degrees(angle)
            
            # Natural arm angles are typically 90-180 degrees
            if 90 <= angle_degrees <= 180:
                return 1.0
            elif 60 <= angle_degrees < 90:
                return (angle_degrees - 60) / 30 * 0.5 + 0.5
            elif angle_degrees < 60:
                return max(0.0, angle_degrees / 60 * 0.5)
            else:
                return 0.5  # Extreme angles
            
        except Exception as e:
            logger.error(f"Erro ao calcular naturalidade do braço: {e}")
            return 0.5
    
    def _analyze_facial_orientation(self, face_points: Optional[np.ndarray], 
                                  pose_points: np.ndarray) -> FacialOrientation:
        """
        Analyze facial orientation (frontal, profile, etc.)
        """
        try:
            if face_points is None or len(face_points) == 0:
                # Use pose landmarks as fallback
                return self._estimate_face_orientation_from_pose(pose_points)
            
            # Use face landmarks for more accurate orientation
            return self._calculate_face_orientation_from_landmarks(face_points)
            
        except Exception as e:
            logger.error(f"Erro ao analisar orientação facial: {e}")
            return FacialOrientation.UNKNOWN
    
    def _estimate_face_orientation_from_pose(self, pose_points: np.ndarray) -> FacialOrientation:
        """
        Estimate face orientation using pose landmarks
        """
        try:
            # Use ear landmarks to estimate orientation
            left_ear, right_ear = pose_points[7], pose_points[8]
            
            # Calculate distance between ears
            ear_distance = np.linalg.norm(left_ear - right_ear)
            
            # Use nose position relative to ears
            nose = pose_points[0]
            
            # Calculate nose position relative to ear midpoint
            ear_midpoint = (left_ear + right_ear) / 2
            nose_offset = nose[0] - ear_midpoint[0]
            
            # Normalize by ear distance
            if ear_distance > 0:
                normalized_offset = abs(nose_offset) / ear_distance
                
                if normalized_offset < 0.1:
                    return FacialOrientation.FRONTAL
                elif normalized_offset < 0.5:
                    return FacialOrientation.THREE_QUARTER
                else:
                    return FacialOrientation.PROFILE
            
            return FacialOrientation.FRONTAL
            
        except Exception as e:
            logger.error(f"Erro ao estimar orientação facial: {e}")
            return FacialOrientation.UNKNOWN
    
    def _calculate_face_orientation_from_landmarks(self, face_points: np.ndarray) -> FacialOrientation:
        """
        Calculate face orientation using detailed face landmarks
        """
        # This would use specific face landmark indices for more accurate orientation
        # For now, return a simplified estimation
        return FacialOrientation.FRONTAL
    
    def _assess_pose_naturalness(self, pose_points: np.ndarray) -> Tuple[PoseNaturalness, float]:
        """
        Assess how natural the pose looks
        """
        try:
            # Calculate various naturalness factors
            joint_angles = self._calculate_joint_angles(pose_points)
            gesture_score = self._analyze_gestures(pose_points)
            balance_score = self._analyze_balance(pose_points)
            
            # Combine scores
            naturalness_score = (
                self._score_joint_naturalness(joint_angles) * 0.4 +
                gesture_score * 0.3 +
                balance_score * 0.3
            )
            
            # Classify naturalness level
            if naturalness_score >= 0.8:
                level = PoseNaturalness.VERY_NATURAL
            elif naturalness_score >= 0.6:
                level = PoseNaturalness.NATURAL
            elif naturalness_score >= 0.4:
                level = PoseNaturalness.SOMEWHAT_NATURAL
            elif naturalness_score >= 0.2:
                level = PoseNaturalness.FORCED
            else:
                level = PoseNaturalness.VERY_FORCED
            
            return level, naturalness_score
            
        except Exception as e:
            logger.error(f"Erro ao avaliar naturalidade da pose: {e}")
            return PoseNaturalness.NATURAL, 0.5
    
    def _calculate_joint_angles(self, pose_points: np.ndarray) -> Dict[str, float]:
        """
        Calculate angles at major joints
        """
        angles = {}
        
        try:
            # Calculate elbow angles
            if len(pose_points) > 16:
                # Left elbow
                left_shoulder, left_elbow, left_wrist = pose_points[11], pose_points[13], pose_points[15]
                angles['left_elbow'] = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
                
                # Right elbow
                right_shoulder, right_elbow, right_wrist = pose_points[12], pose_points[14], pose_points[16]
                angles['right_elbow'] = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
                
                # Knee angles (if available)
                if len(pose_points) > 26:
                    left_hip, left_knee, left_ankle = pose_points[23], pose_points[25], pose_points[27]
                    angles['left_knee'] = self._calculate_angle(left_hip, left_knee, left_ankle)
                    
                    right_hip, right_knee, right_ankle = pose_points[24], pose_points[26], pose_points[28]
                    angles['right_knee'] = self._calculate_angle(right_hip, right_knee, right_ankle)
            
        except Exception as e:
            logger.error(f"Erro ao calcular ângulos das articulações: {e}")
        
        return angles
    
    def _calculate_angle(self, point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
        """
        Calculate angle at point2 formed by point1-point2-point3
        """
        try:
            vector1 = point1 - point2
            vector2 = point3 - point2
            
            dot_product = np.dot(vector1, vector2)
            norms = np.linalg.norm(vector1) * np.linalg.norm(vector2)
            
            if norms == 0:
                return 0.0
            
            cos_angle = dot_product / norms
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = math.acos(cos_angle)
            
            return math.degrees(angle)
            
        except Exception as e:
            logger.error(f"Erro ao calcular ângulo: {e}")
            return 0.0
    
    def _score_joint_naturalness(self, joint_angles: Dict[str, float]) -> float:
        """
        Score how natural the joint angles are
        """
        if not joint_angles:
            return 0.5
        
        scores = []
        
        # Natural angle ranges for different joints
        natural_ranges = {
            'left_elbow': (90, 180),
            'right_elbow': (90, 180),
            'left_knee': (160, 180),
            'right_knee': (160, 180)
        }
        
        for joint, angle in joint_angles.items():
            if joint in natural_ranges:
                min_angle, max_angle = natural_ranges[joint]
                if min_angle <= angle <= max_angle:
                    scores.append(1.0)
                else:
                    # Gradual penalty for angles outside natural range
                    if angle < min_angle:
                        scores.append(max(0.0, angle / min_angle))
                    else:
                        scores.append(max(0.0, (360 - angle) / (360 - max_angle)))
        
        return float(np.mean(scores)) if scores else 0.5
    
    def _analyze_gestures(self, pose_points: np.ndarray) -> float:
        """
        Analyze gesture naturalness
        """
        # Simplified gesture analysis
        # In a full implementation, this would detect specific gestures
        return 0.7  # Default neutral score
    
    def _analyze_balance(self, pose_points: np.ndarray) -> float:
        """
        Analyze body balance and stability
        """
        try:
            # Calculate center of mass approximation
            if len(pose_points) > 24:
                # Use hip and shoulder midpoints
                shoulder_midpoint = (pose_points[11] + pose_points[12]) / 2
                hip_midpoint = (pose_points[23] + pose_points[24]) / 2
                
                # Check if body is balanced
                horizontal_offset = abs(shoulder_midpoint[0] - hip_midpoint[0])
                torso_width = abs(pose_points[11][0] - pose_points[12][0])
                
                if torso_width > 0:
                    balance_ratio = horizontal_offset / torso_width
                    return max(0.0, 1.0 - balance_ratio)
            
            return 0.7
            
        except Exception as e:
            logger.error(f"Erro ao analisar equilíbrio: {e}")
            return 0.5
    
    def _calculate_body_symmetry(self, pose_points: np.ndarray) -> float:
        """
        Calculate body symmetry score
        """
        try:
            if len(pose_points) < 17:
                return 0.5
            
            # Compare left and right side landmarks
            left_shoulder, right_shoulder = pose_points[11], pose_points[12]
            left_elbow, right_elbow = pose_points[13], pose_points[14]
            left_wrist, right_wrist = pose_points[15], pose_points[16]
            
            # Calculate center line
            center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            
            # Calculate symmetry for arms
            left_arm_offset = abs(left_shoulder[0] - center_x)
            right_arm_offset = abs(right_shoulder[0] - center_x)
            
            if left_arm_offset + right_arm_offset > 0:
                symmetry_score = 1.0 - abs(left_arm_offset - right_arm_offset) / (left_arm_offset + right_arm_offset)
            else:
                symmetry_score = 1.0
            
            return max(0.0, min(1.0, symmetry_score))
            
        except Exception as e:
            logger.error(f"Erro ao calcular simetria corporal: {e}")
            return 0.5
    
    def _analyze_motion_and_stability(self, pose_points: np.ndarray) -> Tuple[MotionType, float]:
        """
        Analyze motion and pose stability
        """
        # For single image analysis, we can only detect signs of motion blur
        # In a video context, this would analyze pose stability over time
        
        # Simplified analysis - assume static pose for photos
        motion_type = MotionType.STATIC
        stability_score = 0.8  # Default good stability for photos
        
        return motion_type, stability_score
    
    def _analyze_body_alignment(self, pose_points: np.ndarray) -> Dict[str, float]:
        """
        Analyze overall body alignment
        """
        alignment = {}
        
        try:
            # Calculate spine alignment
            alignment['spine'] = self._calculate_spine_alignment(pose_points)
            
            # Calculate shoulder alignment
            alignment['shoulders'] = self._calculate_shoulder_levelness(pose_points)
            
            # Calculate hip alignment
            alignment['hips'] = self._calculate_hip_alignment(pose_points)
            
        except Exception as e:
            logger.error(f"Erro ao analisar alinhamento corporal: {e}")
        
        return alignment
    
    def _calculate_overall_pose_rating(self, posture_score: float, naturalness_score: float, 
                                     symmetry_score: float, stability_score: float) -> str:
        """
        Calculate overall pose rating
        """
        combined_score = (
            posture_score * self.pose_weights['posture'] +
            naturalness_score * self.pose_weights['naturalness'] +
            symmetry_score * self.pose_weights['symmetry'] +
            stability_score * self.pose_weights['stability']
        )
        
        # Add facial and other weights (default to 0.7 for missing components)
        combined_score += 0.7 * (self.pose_weights['facial'])
        
        if combined_score >= 0.8:
            return "excellent"
        elif combined_score >= 0.6:
            return "good"
        elif combined_score >= 0.4:
            return "acceptable"
        else:
            return "poor"
    
    def _get_default_pose_metrics(self) -> PoseQualityMetrics:
        """
        Return default pose metrics in case of error
        """
        return PoseQualityMetrics(
            posture_quality_score=0.5,
            facial_orientation=FacialOrientation.UNKNOWN,
            naturalness_level=PoseNaturalness.NATURAL,
            motion_type=MotionType.STATIC,
            pose_stability_score=0.5,
            symmetry_score=0.5,
            overall_pose_rating="unknown",
            detailed_analysis={}
        )


if __name__ == "__main__":
    # Example usage and testing
    import sys
    import os
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("🔍 Testando PoseQualityAnalyzer...")
    
    # Mock pose landmarks (33 points for MediaPipe pose)
    # This would normally come from MediaPipe pose detection
    mock_pose_landmarks = []
    for i in range(33):
        # Create mock normalized coordinates
        x = 0.5 + (i % 3 - 1) * 0.1  # Spread around center
        y = 0.3 + (i // 11) * 0.2     # Vertical distribution
        mock_pose_landmarks.append([x, y])
    
    # Test the analyzer
    analyzer = PoseQualityAnalyzer()
    image_shape = (480, 640)  # height, width
    
    pose_metrics = analyzer.analyze_pose_quality(
        mock_pose_landmarks, 
        image_shape=image_shape
    )
    
    print(f"   ✅ Qualidade da postura: {pose_metrics.posture_quality_score:.3f}")
    print(f"   ✅ Orientação facial: {pose_metrics.facial_orientation.value}")
    print(f"   ✅ Naturalidade: {pose_metrics.naturalness_level.value}")
    print(f"   ✅ Tipo de movimento: {pose_metrics.motion_type.value}")
    print(f"   ✅ Estabilidade da pose: {pose_metrics.pose_stability_score:.3f}")
    print(f"   ✅ Simetria corporal: {pose_metrics.symmetry_score:.3f}")
    print(f"   ✅ Avaliação geral: {pose_metrics.overall_pose_rating}")
    
    print("🎉 PoseQualityAnalyzer funcionando!")
