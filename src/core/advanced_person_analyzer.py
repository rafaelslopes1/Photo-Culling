#!/usr/bin/env python3
"""
Advanced Person Analyzer Module for Photo Culling System
Módulo integrador de análise avançada de pessoas para sistema de seleção de fotos

Integrates all Phase 2 person analysis modules into a unified pipeline
Following the Phase 2 roadmap specifications
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, asdict
import json

# Import Phase 2 analyzers
try:
    from .person_quality_analyzer import PersonQualityAnalyzer, PersonQualityMetrics
    from .cropping_analyzer import CroppingAnalyzer, CroppingAnalysis
    from .pose_quality_analyzer import PoseQualityAnalyzer, PoseQualityMetrics
except ImportError:
    # For direct execution
    from person_quality_analyzer import PersonQualityAnalyzer, PersonQualityMetrics
    from cropping_analyzer import CroppingAnalyzer, CroppingAnalysis
    from pose_quality_analyzer import PoseQualityAnalyzer, PoseQualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class AdvancedPersonAnalysis:
    """Comprehensive person analysis results from all Phase 2 modules"""
    # Person quality analysis
    person_quality: PersonQualityMetrics
    
    # Cropping analysis
    cropping_analysis: CroppingAnalysis
    
    # Pose quality analysis
    pose_quality: PoseQualityMetrics
    
    # Combined scores
    overall_person_score: float
    person_rating: str
    
    # Metadata
    analysis_version: str = "2.0"
    processing_timestamp: str = ""


class AdvancedPersonAnalyzer:
    """
    Unified analyzer that combines all Phase 2 person analysis capabilities
    Provides comprehensive analysis of person quality, cropping, and pose
    """
    
    def __init__(self):
        """Initialize all Phase 2 analyzers"""
        try:
            self.person_quality_analyzer = PersonQualityAnalyzer()
            self.cropping_analyzer = CroppingAnalyzer()
            self.pose_quality_analyzer = PoseQualityAnalyzer()
            
            # Weights for combining different analysis types
            self.analysis_weights = {
                'person_quality': 0.4,    # Local blur, lighting, contrast
                'cropping_framing': 0.35, # Cropping issues and framing
                'pose_quality': 0.25      # Pose naturalness and stability
            }
            
            logger.info("AdvancedPersonAnalyzer inicializado com todos os módulos da Fase 2")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar AdvancedPersonAnalyzer: {e}")
            raise
    
    def analyze_person_comprehensive(self, 
                                   person_bbox: Tuple[int, int, int, int],
                                   face_bbox: Optional[Tuple[int, int, int, int]],
                                   pose_landmarks: Optional[List],
                                   face_landmarks: Optional[List],
                                   full_image: np.ndarray,
                                   dominant_person_data: Optional[Dict] = None) -> AdvancedPersonAnalysis:
        """
        Comprehensive analysis of a person using all Phase 2 capabilities
        
        Args:
            person_bbox: Person bounding box (x, y, w, h)
            face_bbox: Face bounding box (x, y, w, h) 
            pose_landmarks: MediaPipe pose landmarks
            face_landmarks: MediaPipe face landmarks
            full_image: Complete image in BGR format
            dominant_person_data: Additional data about the person
            
        Returns:
            AdvancedPersonAnalysis with comprehensive results
        """
        try:
            import datetime
            
            # 1. Person Quality Analysis
            logger.debug("Executando análise de qualidade da pessoa...")
            person_quality = self.person_quality_analyzer.analyze_person_quality(
                person_bbox, full_image, dominant_person_data
            )
            
            # 2. Cropping Analysis
            logger.debug("Executando análise de cortes e enquadramento...")
            height, width = full_image.shape[:2]
            channels = full_image.shape[2] if len(full_image.shape) > 2 else 1
            img_shape = (height, width, channels)
            cropping_analysis = self.cropping_analyzer.analyze_cropping_issues(
                person_bbox, img_shape, face_bbox
            )
            
            # 3. Pose Quality Analysis  
            logger.debug("Executando análise de qualidade de pose...")
            pose_img_shape = (height, width)
            pose_quality = self.pose_quality_analyzer.analyze_pose_quality(
                pose_landmarks, face_landmarks, pose_img_shape
            )
            
            # 4. Calculate combined scores
            overall_score, rating = self._calculate_combined_scores(
                person_quality, cropping_analysis, pose_quality
            )
            
            # 5. Create comprehensive analysis result
            analysis = AdvancedPersonAnalysis(
                person_quality=person_quality,
                cropping_analysis=cropping_analysis,
                pose_quality=pose_quality,
                overall_person_score=overall_score,
                person_rating=rating,
                processing_timestamp=datetime.datetime.now().isoformat()
            )
            
            logger.info(f"Análise avançada completa - Score: {overall_score:.3f}, Rating: {rating}")
            return analysis
            
        except Exception as e:
            logger.error(f"Erro na análise avançada da pessoa: {e}")
            return self._get_default_analysis()
    
    def _calculate_combined_scores(self, 
                                 person_quality: PersonQualityMetrics,
                                 cropping_analysis: CroppingAnalysis,
                                 pose_quality: PoseQualityMetrics) -> Tuple[float, str]:
        """
        Calculate combined overall score and rating from all analyses
        """
        try:
            # Extract individual scores
            quality_score = person_quality.overall_quality
            framing_score = cropping_analysis.framing_quality_score
            pose_score = pose_quality.posture_quality_score
            
            # Apply cropping penalty
            cropping_penalty = self._calculate_cropping_penalty(cropping_analysis)
            
            # Calculate weighted average
            combined_score = (
                quality_score * self.analysis_weights['person_quality'] +
                framing_score * self.analysis_weights['cropping_framing'] +
                pose_score * self.analysis_weights['pose_quality']
            )
            
            # Apply cropping penalty
            final_score = max(0.0, combined_score - cropping_penalty)
            
            # Determine rating
            if final_score >= 0.8:
                rating = "excellent"
            elif final_score >= 0.65:
                rating = "good"
            elif final_score >= 0.45:
                rating = "acceptable"
            elif final_score >= 0.25:
                rating = "poor"
            else:
                rating = "very_poor"
            
            return final_score, rating
            
        except Exception as e:
            logger.error(f"Erro ao calcular scores combinados: {e}")
            return 0.5, "unknown"
    
    def _calculate_cropping_penalty(self, cropping_analysis: CroppingAnalysis) -> float:
        """
        Calculate penalty for cropping issues
        """
        try:
            from cropping_analyzer import CroppingSeverity
        except ImportError:
            from .cropping_analyzer import CroppingSeverity
        
        penalties = {
            CroppingSeverity.NONE: 0.0,
            CroppingSeverity.MINOR: 0.05,
            CroppingSeverity.MODERATE: 0.15,
            CroppingSeverity.SEVERE: 0.3
        }
        
        return penalties.get(cropping_analysis.severity, 0.1)
    
    def extract_features_for_database(self, analysis: AdvancedPersonAnalysis) -> Dict[str, Any]:
        """
        Extract features from comprehensive analysis for database storage
        
        Returns:
            Dictionary with features suitable for FeatureExtractor integration
        """
        try:
            features = {}
            
            # Person Quality Features
            pq = analysis.person_quality
            features.update({
                'person_local_blur_score': pq.local_blur_score,
                'person_lighting_quality': pq.lighting_quality,
                'person_contrast_score': pq.contrast_score,
                'person_relative_sharpness': pq.relative_sharpness,
                'person_quality_score': pq.overall_quality,
                'person_quality_level': pq.quality_level.value
            })
            
            # Cropping Analysis Features
            ca = analysis.cropping_analysis
            features.update({
                'has_cropping_issues': ca.has_cropping_issues,
                'cropping_severity': ca.severity.value,
                'cropping_types': json.dumps([ct.value for ct in ca.cropping_types]),
                'min_edge_distance': ca.edge_distances.get('min_edge', 0.0),
                'framing_quality_score': ca.framing_quality_score,
                'composition_score': ca.composition_score,
                'framing_rating': ca.overall_framing_rating
            })
            
            # Pose Quality Features
            pq_pose = analysis.pose_quality
            features.update({
                'posture_quality_score': pq_pose.posture_quality_score,
                'facial_orientation': pq_pose.facial_orientation.value,
                'pose_naturalness': pq_pose.naturalness_level.value,
                'motion_type': pq_pose.motion_type.value,
                'pose_stability_score': pq_pose.pose_stability_score,
                'body_symmetry_score': pq_pose.symmetry_score,
                'pose_rating': pq_pose.overall_pose_rating
            })
            
            # Combined Features
            features.update({
                'overall_person_score': analysis.overall_person_score,
                'overall_person_rating': analysis.person_rating,
                'advanced_analysis_version': analysis.analysis_version
            })
            
            return features
            
        except Exception as e:
            logger.error(f"Erro ao extrair features para banco de dados: {e}")
            return {}
    
    def generate_analysis_report(self, analysis: AdvancedPersonAnalysis) -> Dict[str, Any]:
        """
        Generate human-readable analysis report
        """
        try:
            report = {
                'summary': {
                    'overall_score': analysis.overall_person_score,
                    'overall_rating': analysis.person_rating,
                    'timestamp': analysis.processing_timestamp
                },
                'person_quality': {
                    'score': analysis.person_quality.overall_quality,
                    'level': analysis.person_quality.quality_level.value,
                    'local_blur': analysis.person_quality.local_blur_score,
                    'lighting': analysis.person_quality.lighting_quality,
                    'contrast': analysis.person_quality.contrast_score,
                    'relative_sharpness': analysis.person_quality.relative_sharpness
                },
                'framing_analysis': {
                    'has_issues': analysis.cropping_analysis.has_cropping_issues,
                    'severity': analysis.cropping_analysis.severity.value,
                    'issues': [ct.value for ct in analysis.cropping_analysis.cropping_types],
                    'framing_score': analysis.cropping_analysis.framing_quality_score,
                    'composition_score': analysis.cropping_analysis.composition_score,
                    'rating': analysis.cropping_analysis.overall_framing_rating
                },
                'pose_analysis': {
                    'posture_score': analysis.pose_quality.posture_quality_score,
                    'naturalness': analysis.pose_quality.naturalness_level.value,
                    'facial_orientation': analysis.pose_quality.facial_orientation.value,
                    'stability': analysis.pose_quality.pose_stability_score,
                    'symmetry': analysis.pose_quality.symmetry_score,
                    'rating': analysis.pose_quality.overall_pose_rating
                },
                'recommendations': self._generate_recommendations(analysis)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório de análise: {e}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, analysis: AdvancedPersonAnalysis) -> List[str]:
        """
        Generate actionable recommendations based on analysis
        """
        recommendations = []
        
        try:
            # Person quality recommendations
            pq = analysis.person_quality
            if pq.local_blur_score < 0.5:
                recommendations.append("Pessoa apresenta blur local - considere fotos mais nítidas")
            
            if pq.lighting_quality < 0.4:
                recommendations.append("Iluminação inadequada na pessoa - verificar exposição")
            
            if pq.relative_sharpness < 0.3:
                recommendations.append("Pessoa menos nítida que o fundo - problema de foco")
            
            # Cropping recommendations
            ca = analysis.cropping_analysis
            if ca.has_cropping_issues:
                if ca.severity.value == "severe":
                    recommendations.append("Problemas severos de corte - rejeitar ou recortar")
                else:
                    recommendations.append(f"Problemas de corte {ca.severity.value} detectados")
            
            if ca.framing_quality_score < 0.5:
                recommendations.append("Enquadramento pode ser melhorado")
            
            # Pose recommendations
            pq_pose = analysis.pose_quality
            if pq_pose.naturalness_level.value in ["forced", "very_forced"]:
                recommendations.append("Pose parece forçada - considere poses mais naturais")
            
            if pq_pose.posture_quality_score < 0.4:
                recommendations.append("Problemas de postura detectados")
            
            # Overall recommendations
            if analysis.overall_person_score < 0.3:
                recommendations.append("Qualidade geral baixa - considere rejeitar esta imagem")
            elif analysis.overall_person_score > 0.8:
                recommendations.append("Excelente qualidade geral - imagem recomendada")
            
        except Exception as e:
            logger.error(f"Erro ao gerar recomendações: {e}")
            recommendations.append("Erro ao gerar recomendações")
        
        return recommendations
    
    def _get_default_analysis(self) -> AdvancedPersonAnalysis:
        """
        Return default analysis in case of error
        """
        return AdvancedPersonAnalysis(
            person_quality=self.person_quality_analyzer._get_default_quality_metrics(),
            cropping_analysis=self.cropping_analyzer._get_default_cropping_analysis(),
            pose_quality=self.pose_quality_analyzer._get_default_pose_metrics(),
            overall_person_score=0.0,
            person_rating="unknown"
        )


def analyze_person_batch_advanced(image_paths: List[str],
                                person_bboxes: List[Tuple[int, int, int, int]],
                                face_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
                                pose_landmarks_list: Optional[List] = None) -> List[AdvancedPersonAnalysis]:
    """
    Batch advanced person analysis for multiple images
    
    Args:
        image_paths: List of image file paths
        person_bboxes: List of person bounding boxes
        face_bboxes: Optional list of face bounding boxes
        pose_landmarks_list: Optional list of pose landmarks
        
    Returns:
        List of AdvancedPersonAnalysis results
    """
    analyzer = AdvancedPersonAnalyzer()
    results = []
    
    for i, (image_path, person_bbox) in enumerate(zip(image_paths, person_bboxes)):
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Não foi possível carregar imagem: {image_path}")
                results.append(analyzer._get_default_analysis())
                continue
            
            # Get corresponding data
            face_bbox = face_bboxes[i] if face_bboxes and i < len(face_bboxes) else None
            pose_landmarks = pose_landmarks_list[i] if pose_landmarks_list and i < len(pose_landmarks_list) else None
            
            # Perform comprehensive analysis
            analysis = analyzer.analyze_person_comprehensive(
                person_bbox=person_bbox,
                face_bbox=face_bbox,
                pose_landmarks=pose_landmarks,
                face_landmarks=None,  # Would need separate face landmarks list
                full_image=image
            )
            
            results.append(analysis)
            
        except Exception as e:
            logger.error(f"Erro ao processar {image_path}: {e}")
            results.append(analyzer._get_default_analysis())
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    import sys
    import os
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with a sample image
    test_image_path = "../../data/input/IMG_0001.JPG"
    if os.path.exists(test_image_path):
        print("🔍 Testando AdvancedPersonAnalyzer...")
        
        # Load test image
        test_image = cv2.imread(test_image_path)
        
        # Mock test data (would come from PersonDetector)
        test_person_bbox = (800, 400, 600, 1000)  # x, y, w, h
        test_face_bbox = (950, 450, 200, 250)     # x, y, w, h
        
        # Mock pose landmarks (would come from MediaPipe)
        mock_pose_landmarks = []
        for i in range(33):
            x = 0.5 + (i % 3 - 1) * 0.1
            y = 0.3 + (i // 11) * 0.2
            mock_pose_landmarks.append([x, y])
        
        # Perform comprehensive analysis
        analyzer = AdvancedPersonAnalyzer()
        analysis = analyzer.analyze_person_comprehensive(
            person_bbox=test_person_bbox,
            face_bbox=test_face_bbox,
            pose_landmarks=mock_pose_landmarks,
            face_landmarks=None,
            full_image=test_image
        )
        
        # Display results
        print(f"   ✅ Score geral da pessoa: {analysis.overall_person_score:.3f}")
        print(f"   ✅ Rating geral: {analysis.person_rating}")
        print(f"   ✅ Qualidade da pessoa: {analysis.person_quality.quality_level.value}")
        print(f"   ✅ Problemas de corte: {analysis.cropping_analysis.has_cropping_issues}")
        print(f"   ✅ Naturalidade da pose: {analysis.pose_quality.naturalness_level.value}")
        
        # Generate report
        report = analyzer.generate_analysis_report(analysis)
        print(f"   ✅ Recomendações: {len(report['recommendations'])} itens")
        
        # Extract features for database
        features = analyzer.extract_features_for_database(analysis)
        print(f"   ✅ Features extraídas: {len(features)} campos")
        
        print("🎉 AdvancedPersonAnalyzer funcionando!")
    else:
        print(f"❌ Imagem de teste não encontrada: {test_image_path}")
