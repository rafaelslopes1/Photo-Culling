"""
Overexposure Analyzer for Photo Culling System v2.5
Analyzes localized overexposure in person regions, specifically for sports photography
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class OverexposureAnalyzer:
    """
    Analyzes localized overexposure in person regions with focus on face and torso areas.
    Specifically designed for sports photography with flash/strong lighting conditions.
    """
    
    def __init__(self):
        # Thresholds optimized for sports photography
        self.critical_threshold = 240  # Pixel value for critical overexposure
        self.moderate_threshold = 220  # Pixel value for moderate overexposure
        
        # Critical ratios for different body regions - ADJUSTED FOR SPORTS PHOTOGRAPHY
        # Based on IMG_0001.JPG analysis: Face 16%, Torso 28% should be considered critical
        self.face_critical_ratio = 0.15   # 15% of face = critical (reduced from 30%)
        self.face_moderate_ratio = 0.08   # 8% of face = moderate concern
        self.torso_critical_ratio = 0.25  # 25% of torso = critical (reduced from 40%)
        self.torso_moderate_ratio = 0.12  # 12% of torso = moderate concern
        
        # Recovery difficulty assessment - ADJUSTED FOR FLASH PHOTOGRAPHY
        self.recovery_thresholds = {
            'easy': 0.08,      # <8% overexposed
            'moderate': 0.20,   # 8-20% overexposed  
            'hard': 0.35,      # 20-35% overexposed
            'impossible': 1.0   # >35% overexposed
        }
    
    def analyze_person_overexposure(self, 
                                  person_bbox: Tuple[int, int, int, int],
                                  face_landmarks: Optional[np.ndarray],
                                  full_image: np.ndarray) -> Dict:
        """
        Analyze overexposure specifically in person regions
        
        Args:
            person_bbox: (x, y, width, height) of person bounding box
            face_landmarks: MediaPipe face landmarks if available
            full_image: Full image as numpy array
            
        Returns:
            Dictionary with overexposure analysis results
        """
        try:
            # Extract person ROI
            person_roi = self._extract_person_roi(person_bbox, full_image)
            
            # Extract face ROI (priority area)
            face_roi = self._extract_face_roi(person_bbox, face_landmarks, full_image)
            
            # Extract torso ROI (secondary priority)
            torso_roi = self._extract_torso_roi(person_bbox, full_image)
            
            # Analyze each region
            face_analysis = self._analyze_region_overexposure(face_roi, "face")
            torso_analysis = self._analyze_region_overexposure(torso_roi, "torso")
            person_overall = self._analyze_region_overexposure(person_roi, "person")
            
            # Combine results and assess severity
            result = self._combine_overexposure_analysis(
                face_analysis, torso_analysis, person_overall
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na análise de superexposição: {e}")
            return self._get_default_overexposure_result()
    
    def _extract_person_roi(self, bbox: Tuple[int, int, int, int], 
                           image: np.ndarray) -> np.ndarray:
        """Extract person region of interest from image"""
        x, y, w, h = bbox
        # Ensure coordinates are within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        return image[y:y+h, x:x+w]
    
    def _extract_face_roi(self, person_bbox: Tuple[int, int, int, int],
                         face_landmarks: Optional[np.ndarray],
                         image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face region with priority for overexposure analysis"""
        if face_landmarks is None:
            # Estimate face area as top 25% of person bbox
            x, y, w, h = person_bbox
            face_h = int(h * 0.25)
            return image[y:y+face_h, x:x+w]
        
        # Use actual face landmarks if available
        # TODO: Implement precise face ROI extraction from landmarks
        return self._extract_face_from_landmarks(face_landmarks, image)
    
    def _extract_torso_roi(self, person_bbox: Tuple[int, int, int, int],
                          image: np.ndarray) -> np.ndarray:
        """Extract torso region (chest/upper body area)"""
        x, y, w, h = person_bbox
        
        # Torso typically occupies middle 40% of person height
        torso_start_y = int(h * 0.25)  # Start after head area
        torso_height = int(h * 0.40)   # 40% of total height
        
        return image[y+torso_start_y:y+torso_start_y+torso_height, x:x+w]
    
    def _analyze_region_overexposure(self, roi: Optional[np.ndarray], 
                                   region_name: str) -> Dict:
        """Analyze overexposure in a specific region"""
        if roi is None or roi.size == 0:
            return {
                'region': region_name,
                'overexposed_ratio_critical': 0.0,
                'overexposed_ratio_moderate': 0.0,
                'mean_brightness': 0.0,
                'max_brightness': 0.0,
                'is_critical': False,
                'is_moderate': False
            }
        
        # Convert to grayscale for brightness analysis
        if len(roi.shape) == 3:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi
        
        total_pixels = gray_roi.size
        
        # Count overexposed pixels
        critical_pixels = np.sum(gray_roi >= self.critical_threshold)
        moderate_pixels = np.sum(gray_roi >= self.moderate_threshold)
        
        # Calculate ratios
        critical_ratio = critical_pixels / total_pixels
        moderate_ratio = moderate_pixels / total_pixels
        
        # Determine severity based on region type
        if region_name == "face":
            is_critical = critical_ratio > self.face_critical_ratio
            is_moderate = moderate_ratio > self.face_moderate_ratio
        elif region_name == "torso":
            is_critical = critical_ratio > self.torso_critical_ratio
            is_moderate = moderate_ratio > self.torso_moderate_ratio
        else:
            # Generic thresholds for overall person
            is_critical = critical_ratio > 0.25
            is_moderate = moderate_ratio > 0.15
        
        return {
            'region': region_name,
            'overexposed_ratio_critical': critical_ratio,
            'overexposed_ratio_moderate': moderate_ratio,
            'mean_brightness': float(np.mean(gray_roi)),
            'max_brightness': float(np.max(gray_roi)),
            'is_critical': is_critical,
            'is_moderate': is_moderate
        }
    
    def _combine_overexposure_analysis(self, face_analysis: Dict, 
                                     torso_analysis: Dict,
                                     person_analysis: Dict) -> Dict:
        """Combine individual region analyses into overall assessment"""
        
        # Overall critical assessment
        overall_critical = (
            face_analysis['is_critical'] or 
            torso_analysis['is_critical']
        )
        
        # Overall moderate assessment  
        overall_moderate = (
            face_analysis['is_moderate'] or 
            torso_analysis['is_moderate']
        )
        
        # Recovery difficulty assessment
        max_critical_ratio = max(
            face_analysis['overexposed_ratio_critical'],
            torso_analysis['overexposed_ratio_critical']
        )
        
        recovery_difficulty = self._assess_recovery_difficulty(max_critical_ratio)
        
        # Primary rejection reason
        main_reason = self._determine_main_overexposure_reason(
            face_analysis, torso_analysis
        )
        
        return {
            # Individual region results
            'face_overexposed_ratio': face_analysis['overexposed_ratio_critical'],
            'torso_overexposed_ratio': torso_analysis['overexposed_ratio_critical'],
            'face_mean_brightness': face_analysis['mean_brightness'],
            'torso_mean_brightness': torso_analysis['mean_brightness'],
            
            # Critical assessments
            'face_critical_overexposure': face_analysis['is_critical'],
            'torso_critical_overexposure': torso_analysis['is_critical'],
            'overall_critical_overexposure': overall_critical,
            
            # Moderate assessments
            'face_moderate_overexposure': face_analysis['is_moderate'],
            'torso_moderate_overexposure': torso_analysis['is_moderate'],
            'overall_moderate_overexposure': overall_moderate,
            
            # Recovery and severity
            'recovery_difficulty': recovery_difficulty,
            'main_overexposure_reason': main_reason,
            'max_overexposed_ratio': max_critical_ratio,
            
            # Recommendation
            'recommendation': self._generate_overexposure_recommendation(
                overall_critical, overall_moderate, recovery_difficulty
            )
        }
    
    def _assess_recovery_difficulty(self, max_overexposed_ratio: float) -> str:
        """Assess difficulty of recovering overexposed areas in post-processing"""
        if max_overexposed_ratio <= self.recovery_thresholds['easy']:
            return 'easy'
        elif max_overexposed_ratio <= self.recovery_thresholds['moderate']:
            return 'moderate'
        elif max_overexposed_ratio <= self.recovery_thresholds['hard']:
            return 'hard'
        else:
            return 'impossible'
    
    def _determine_main_overexposure_reason(self, face_analysis: Dict, 
                                          torso_analysis: Dict) -> str:
        """Determine the primary reason for overexposure issues"""
        if face_analysis['is_critical']:
            return 'face_critical_overexposure'
        elif torso_analysis['is_critical']:
            return 'torso_critical_overexposure'
        elif face_analysis['is_moderate']:
            return 'face_moderate_overexposure'
        elif torso_analysis['is_moderate']:
            return 'torso_moderate_overexposure'
        else:
            return 'no_significant_overexposure'
    
    def _generate_overexposure_recommendation(self, is_critical: bool, 
                                            is_moderate: bool,
                                            recovery_difficulty: str) -> str:
        """Generate recommendation based on overexposure analysis"""
        if is_critical:
            if recovery_difficulty == 'impossible':
                return 'reject_unrecoverable_overexposure'
            elif recovery_difficulty == 'hard':
                return 'review_difficult_recovery'
            else:
                return 'review_recoverable_overexposure'
        elif is_moderate:
            return 'acceptable_minor_overexposure'
        else:
            return 'no_overexposure_issues'
    
    def _extract_face_from_landmarks(self, landmarks: np.ndarray, 
                                   image: np.ndarray) -> np.ndarray:
        """Extract face ROI using MediaPipe landmarks (placeholder)"""
        # TODO: Implement precise face extraction using landmarks
        # For now, return a basic face region estimation
        return image  # Placeholder implementation
    
    def _get_default_overexposure_result(self) -> Dict:
        """Return default result in case of analysis failure"""
        return {
            'face_overexposed_ratio': 0.0,
            'torso_overexposed_ratio': 0.0,
            'face_mean_brightness': 0.0,
            'torso_mean_brightness': 0.0,
            'face_critical_overexposure': False,
            'torso_critical_overexposure': False,
            'overall_critical_overexposure': False,
            'face_moderate_overexposure': False,
            'torso_moderate_overexposure': False,
            'overall_moderate_overexposure': False,
            'recovery_difficulty': 'unknown',
            'main_overexposure_reason': 'analysis_failed',
            'max_overexposed_ratio': 0.0,
            'recommendation': 'manual_review_required'
        }

# Test function for IMG_0001.JPG analysis
def test_img_0001_overexposure():
    """
    Test function specifically for IMG_0001.JPG case
    """
    analyzer = OverexposureAnalyzer()
    
    # This would be called with actual image data and person detection results
    print("OverexposureAnalyzer initialized for sports photography")
    print(f"Critical threshold: {analyzer.critical_threshold}")
    print(f"Face critical ratio: {analyzer.face_critical_ratio}")
    print(f"Torso critical ratio: {analyzer.torso_critical_ratio}")
    
    return analyzer

if __name__ == "__main__":
    test_img_0001_overexposure()
