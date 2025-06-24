#!/usr/bin/env python3
"""
Cropping Analyzer Module for Photo Culling System
Módulo de análise de cortes e enquadramento para sistema de seleção de fotos

Detects cropping issues and framing problems for people in images
Following the Phase 2 roadmap specifications
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class CroppingSeverity(Enum):
    """Severity levels for cropping issues"""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class CroppingType(Enum):
    """Types of cropping issues"""
    HEAD_CUT = "head_cut"
    BODY_CUT = "body_cut"
    LIMBS_CUT = "limbs_cut"
    FACE_PARTIAL = "face_partial"
    MULTIPLE_CUTS = "multiple_cuts"


@dataclass
class CroppingAnalysis:
    """Data class for cropping analysis results"""
    has_cropping_issues: bool
    severity: CroppingSeverity
    cropping_types: List[CroppingType]
    edge_distances: Dict[str, float]  # Distance to each edge
    framing_quality_score: float
    composition_score: float
    overall_framing_rating: str


class CroppingAnalyzer:
    """
    Analyzer for detecting cropping issues and framing problems
    Focuses on how well people are positioned and framed in images
    """
    
    def __init__(self):
        """Initialize the cropping analyzer"""
        # Distance thresholds (in pixels) for detecting edge proximity
        self.edge_tolerance = {
            'minor': 20,     # 20px from edge = minor issue
            'moderate': 10,  # 10px from edge = moderate issue  
            'severe': 5      # 5px from edge = severe issue
        }
        
        # Weights for different aspects of framing
        self.framing_weights = {
            'edge_distance': 0.4,      # How far from edges
            'rule_of_thirds': 0.3,     # Positioning according to rule of thirds
            'central_placement': 0.2,   # Not too centered, not too off-center
            'aspect_ratio': 0.1        # Person aspect ratio considerations
        }
    
    def analyze_cropping_issues(self, person_bbox: Tuple[int, int, int, int], 
                              image_shape: Tuple[int, int, int], 
                              face_bbox: Optional[Tuple[int, int, int, int]] = None) -> CroppingAnalysis:
        """
        Comprehensive analysis of cropping and framing issues
        
        Args:
            person_bbox: Bounding box of the person (x, y, w, h)
            image_shape: Shape of the image (height, width, channels)
            face_bbox: Optional face bounding box for more detailed analysis
            
        Returns:
            CroppingAnalysis with detailed results
        """
        try:
            img_height, img_width = image_shape[:2]
            x, y, w, h = person_bbox
            
            # Calculate distances to each edge
            edge_distances = self._calculate_edge_distances(person_bbox, img_width, img_height)
            
            # Detect specific cropping issues
            cropping_issues = self._detect_cropping_types(
                person_bbox, edge_distances, face_bbox
            )
            
            # Determine severity level
            severity = self._classify_cropping_severity(edge_distances, cropping_issues)
            
            # Analyze framing quality
            framing_quality = self._analyze_framing_quality(person_bbox, img_width, img_height)
            
            # Analyze composition (rule of thirds, etc.)
            composition_score = self._analyze_composition(person_bbox, img_width, img_height)
            
            # Overall framing rating
            overall_rating = self._calculate_overall_framing_rating(
                framing_quality, composition_score, severity
            )
            
            return CroppingAnalysis(
                has_cropping_issues=(severity != CroppingSeverity.NONE),
                severity=severity,
                cropping_types=cropping_issues,
                edge_distances=edge_distances,
                framing_quality_score=framing_quality,
                composition_score=composition_score,
                overall_framing_rating=overall_rating
            )
            
        except Exception as e:
            logger.error(f"Erro na análise de cortes: {e}")
            return self._get_default_cropping_analysis()
    
    def _calculate_edge_distances(self, person_bbox: Tuple[int, int, int, int], 
                                img_width: int, img_height: int) -> Dict[str, float]:
        """
        Calculate distances from person bounding box to image edges
        """
        x, y, w, h = person_bbox
        
        # Calculate actual distances to edges
        distances = {
            'left': float(x),
            'right': float(img_width - (x + w)),
            'top': float(y),
            'bottom': float(img_height - (y + h))
        }
        
        # Calculate minimum distance to any edge
        distances['min_edge'] = min(distances['left'], distances['right'], 
                                  distances['top'], distances['bottom'])
        
        return distances
    
    def _detect_cropping_types(self, person_bbox: Tuple[int, int, int, int], 
                             edge_distances: Dict[str, float], 
                             face_bbox: Optional[Tuple[int, int, int, int]]) -> List[CroppingType]:
        """
        Detect specific types of cropping issues
        """
        cropping_types = []
        
        # Check for severe edge proximity (likely cropping)
        severe_threshold = self.edge_tolerance['severe']
        moderate_threshold = self.edge_tolerance['moderate']
        
        cuts_detected = []
        
        # Check each edge
        if edge_distances['top'] <= severe_threshold:
            cuts_detected.append('top')
        if edge_distances['bottom'] <= severe_threshold:
            cuts_detected.append('bottom')
        if edge_distances['left'] <= severe_threshold:
            cuts_detected.append('left')
        if edge_distances['right'] <= severe_threshold:
            cuts_detected.append('right')
        
        # Classify based on which edges are cut
        if len(cuts_detected) >= 2:
            cropping_types.append(CroppingType.MULTIPLE_CUTS)
        elif 'top' in cuts_detected:
            # Top cut likely means head is cut
            cropping_types.append(CroppingType.HEAD_CUT)
        elif 'bottom' in cuts_detected:
            # Bottom cut likely means body/limbs are cut
            cropping_types.append(CroppingType.LIMBS_CUT)
        elif 'left' in cuts_detected or 'right' in cuts_detected:
            # Side cuts could be body or limbs
            cropping_types.append(CroppingType.BODY_CUT)
        
        # Special check for face cropping if face_bbox is provided
        if face_bbox and self._is_face_cropped(face_bbox, edge_distances):
            cropping_types.append(CroppingType.FACE_PARTIAL)
        
        return cropping_types
    
    def _is_face_cropped(self, face_bbox: Tuple[int, int, int, int], 
                        edge_distances: Dict[str, float]) -> bool:
        """
        Check if the face specifically is cropped
        """
        fx, fy, fw, fh = face_bbox
        
        # Calculate face distances to edges
        face_edge_distances = {
            'left': fx,
            'right': edge_distances['right'] - (fx + fw - (fx + fw)),  # Simplified
            'top': fy,
            'bottom': edge_distances['bottom'] - (fy + fh - (fy + fh))  # Simplified
        }
        
        # Check if face is very close to any edge
        face_tolerance = 15  # Faces need more space
        
        return any(dist <= face_tolerance for dist in face_edge_distances.values())
    
    def _classify_cropping_severity(self, edge_distances: Dict[str, float], 
                                  cropping_types: List[CroppingType]) -> CroppingSeverity:
        """
        Classify the overall severity of cropping issues
        """
        min_distance = edge_distances['min_edge']
        
        # No cropping issues
        if min_distance > self.edge_tolerance['minor'] and not cropping_types:
            return CroppingSeverity.NONE
        
        # Severe cropping
        if (min_distance <= self.edge_tolerance['severe'] or 
            CroppingType.MULTIPLE_CUTS in cropping_types or
            CroppingType.FACE_PARTIAL in cropping_types):
            return CroppingSeverity.SEVERE
        
        # Moderate cropping
        if (min_distance <= self.edge_tolerance['moderate'] or 
            CroppingType.HEAD_CUT in cropping_types):
            return CroppingSeverity.MODERATE
        
        # Minor cropping
        if min_distance <= self.edge_tolerance['minor'] or cropping_types:
            return CroppingSeverity.MINOR
        
        return CroppingSeverity.NONE
    
    def _analyze_framing_quality(self, person_bbox: Tuple[int, int, int, int], 
                               img_width: int, img_height: int) -> float:
        """
        Analyze overall framing quality of the person in the image
        """
        try:
            x, y, w, h = person_bbox
            
            # Calculate person center
            person_center_x = x + w / 2
            person_center_y = y + h / 2
            
            # Calculate image center
            img_center_x = img_width / 2
            img_center_y = img_height / 2
            
            # 1. Edge distance score (higher = better, further from edges)
            edge_distances = self._calculate_edge_distances(person_bbox, img_width, img_height)
            min_edge_distance = edge_distances['min_edge']
            
            # Normalize edge distance (ideal is around 10% of image dimension)
            ideal_margin = min(img_width, img_height) * 0.1
            edge_score = min(1.0, min_edge_distance / ideal_margin)
            
            # 2. Rule of thirds score
            thirds_score = self._calculate_rule_of_thirds_score(
                person_center_x, person_center_y, img_width, img_height
            )
            
            # 3. Central placement score (not too centered, not too off-center)
            center_distance = np.sqrt(
                (person_center_x - img_center_x) ** 2 + 
                (person_center_y - img_center_y) ** 2
            )
            max_distance = np.sqrt((img_width / 2) ** 2 + (img_height / 2) ** 2)
            
            # Ideal is around 20-40% from center
            normalized_center_distance = center_distance / max_distance
            if 0.2 <= normalized_center_distance <= 0.4:
                central_score = 1.0
            elif normalized_center_distance < 0.2:
                # Too centered
                central_score = normalized_center_distance / 0.2
            else:
                # Too off-center
                central_score = max(0.0, (1.0 - normalized_center_distance) / 0.6)
            
            # 4. Aspect ratio score (person should have reasonable proportions)
            person_aspect_ratio = w / h
            # Typical person aspect ratios range from 0.3 to 0.8
            if 0.3 <= person_aspect_ratio <= 0.8:
                aspect_score = 1.0
            else:
                # Penalize extreme aspect ratios
                aspect_score = max(0.0, 1.0 - abs(person_aspect_ratio - 0.55) / 0.45)
            
            # Combine scores with weights
            framing_quality = (
                edge_score * self.framing_weights['edge_distance'] +
                thirds_score * self.framing_weights['rule_of_thirds'] +
                central_score * self.framing_weights['central_placement'] +
                aspect_score * self.framing_weights['aspect_ratio']
            )
            
            return max(0.0, min(1.0, framing_quality))
            
        except Exception as e:
            logger.error(f"Erro ao analisar qualidade do enquadramento: {e}")
            return 0.5
    
    def _calculate_rule_of_thirds_score(self, person_center_x: float, person_center_y: float, 
                                      img_width: int, img_height: int) -> float:
        """
        Calculate how well the person is positioned according to rule of thirds
        """
        # Rule of thirds lines
        third_x1, third_x2 = img_width / 3, 2 * img_width / 3
        third_y1, third_y2 = img_height / 3, 2 * img_height / 3
        
        # Calculate distances to rule of thirds lines
        x_distances = [
            abs(person_center_x - third_x1),
            abs(person_center_x - third_x2)
        ]
        y_distances = [
            abs(person_center_y - third_y1),
            abs(person_center_y - third_y2)
        ]
        
        # Find minimum distances
        min_x_distance = min(x_distances)
        min_y_distance = min(y_distances)
        
        # Normalize distances (closer to third lines = higher score)
        x_score = max(0.0, 1.0 - min_x_distance / (img_width / 6))  # Within 1/6 of image width
        y_score = max(0.0, 1.0 - min_y_distance / (img_height / 6))  # Within 1/6 of image height
        
        # Combine scores (both x and y positioning matter)
        return (x_score + y_score) / 2
    
    def _analyze_composition(self, person_bbox: Tuple[int, int, int, int], 
                           img_width: int, img_height: int) -> float:
        """
        Analyze compositional aspects beyond basic framing
        """
        try:
            x, y, w, h = person_bbox
            
            # 1. Size appropriateness (person should occupy reasonable portion of image)
            person_area = w * h
            image_area = img_width * img_height
            area_ratio = person_area / image_area
            
            # Ideal range: person occupies 10-50% of image
            if 0.1 <= area_ratio <= 0.5:
                size_score = 1.0
            elif area_ratio < 0.1:
                # Too small
                size_score = area_ratio / 0.1
            else:
                # Too large
                size_score = max(0.0, (1.0 - area_ratio) / 0.5)
            
            # 2. Headroom analysis (space above person's head)
            headroom = y  # Distance from top of image to top of person
            ideal_headroom = img_height * 0.1  # 10% of image height
            
            if headroom >= ideal_headroom:
                headroom_score = 1.0
            else:
                headroom_score = headroom / ideal_headroom
            
            # 3. Standing room (space below person)
            standing_room = img_height - (y + h)
            ideal_standing_room = img_height * 0.05  # 5% of image height
            
            if standing_room >= ideal_standing_room:
                standing_score = 1.0
            else:
                standing_score = standing_room / ideal_standing_room
            
            # Combine composition scores
            composition_score = (size_score * 0.5 + headroom_score * 0.3 + standing_score * 0.2)
            
            return max(0.0, min(1.0, composition_score))
            
        except Exception as e:
            logger.error(f"Erro ao analisar composição: {e}")
            return 0.5
    
    def _calculate_overall_framing_rating(self, framing_quality: float, 
                                        composition_score: float, 
                                        severity: CroppingSeverity) -> str:
        """
        Calculate overall framing rating combining all factors
        """
        # Combine framing and composition
        combined_score = (framing_quality * 0.6 + composition_score * 0.4)
        
        # Apply severity penalty
        severity_penalties = {
            CroppingSeverity.NONE: 0.0,
            CroppingSeverity.MINOR: 0.1,
            CroppingSeverity.MODERATE: 0.3,
            CroppingSeverity.SEVERE: 0.5
        }
        
        final_score = combined_score - severity_penalties[severity]
        final_score = max(0.0, final_score)
        
        # Classify rating
        if final_score >= 0.8:
            return "excellent"
        elif final_score >= 0.6:
            return "good"
        elif final_score >= 0.4:
            return "acceptable"
        else:
            return "poor"
    
    def _get_default_cropping_analysis(self) -> CroppingAnalysis:
        """
        Return default analysis in case of error
        """
        return CroppingAnalysis(
            has_cropping_issues=False,
            severity=CroppingSeverity.NONE,
            cropping_types=[],
            edge_distances={'left': 0, 'right': 0, 'top': 0, 'bottom': 0, 'min_edge': 0},
            framing_quality_score=0.0,
            composition_score=0.0,
            overall_framing_rating="unknown"
        )


def analyze_cropping_batch(image_shapes: List[Tuple[int, int, int]], 
                         person_bboxes: List[Tuple[int, int, int, int]], 
                         face_bboxes: Optional[List[Tuple[int, int, int, int]]] = None) -> List[CroppingAnalysis]:
    """
    Batch analysis of cropping issues for multiple images
    
    Args:
        image_shapes: List of image shapes (height, width, channels)
        person_bboxes: List of person bounding boxes
        face_bboxes: Optional list of face bounding boxes
        
    Returns:
        List of CroppingAnalysis for each image
    """
    analyzer = CroppingAnalyzer()
    results = []
    
    for i, (shape, person_bbox) in enumerate(zip(image_shapes, person_bboxes)):
        try:
            face_bbox = face_bboxes[i] if face_bboxes and i < len(face_bboxes) else None
            analysis = analyzer.analyze_cropping_issues(person_bbox, shape, face_bbox)
            results.append(analysis)
        except Exception as e:
            logger.error(f"Erro ao processar imagem {i}: {e}")
            results.append(analyzer._get_default_cropping_analysis())
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    import sys
    import os
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("🔍 Testando CroppingAnalyzer...")
    
    # Mock test data
    test_image_shape = (2400, 1600, 3)  # height, width, channels
    test_person_bbox = (800, 400, 600, 1000)  # x, y, w, h - person in middle
    test_face_bbox = (950, 450, 200, 250)     # x, y, w, h - face within person
    
    # Test well-framed person
    analyzer = CroppingAnalyzer()
    analysis = analyzer.analyze_cropping_issues(test_person_bbox, test_image_shape, test_face_bbox)
    
    print(f"   ✅ Tem problemas de corte: {analysis.has_cropping_issues}")
    print(f"   ✅ Severidade: {analysis.severity.value}")
    print(f"   ✅ Tipos de corte: {[ct.value for ct in analysis.cropping_types]}")
    print(f"   ✅ Distância mínima da borda: {analysis.edge_distances['min_edge']:.1f}px")
    print(f"   ✅ Qualidade do enquadramento: {analysis.framing_quality_score:.3f}")
    print(f"   ✅ Score de composição: {analysis.composition_score:.3f}")
    print(f"   ✅ Avaliação geral: {analysis.overall_framing_rating}")
    
    # Test cropped person (close to edge)
    print("\n🔍 Testando pessoa cortada...")
    cropped_person_bbox = (5, 10, 600, 1000)  # Very close to left and top edges
    cropped_analysis = analyzer.analyze_cropping_issues(cropped_person_bbox, test_image_shape)
    
    print(f"   ✅ Tem problemas de corte: {cropped_analysis.has_cropping_issues}")
    print(f"   ✅ Severidade: {cropped_analysis.severity.value}")
    print(f"   ✅ Tipos de corte: {[ct.value for ct in cropped_analysis.cropping_types]}")
    print(f"   ✅ Avaliação geral: {cropped_analysis.overall_framing_rating}")
    
    print("🎉 CroppingAnalyzer funcionando!")
