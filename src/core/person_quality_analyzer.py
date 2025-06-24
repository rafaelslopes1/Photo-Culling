#!/usr/bin/env python3
"""
Person Quality Analyzer Module for Photo Culling System
Módulo de análise de qualidade específica da pessoa para sistema de seleção de fotos

Implements advanced quality analysis focused on the dominant person in the image
Following the Phase 2 roadmap specifications
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PersonQualityLevel(Enum):
    """Quality levels for person-specific analysis"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


@dataclass
class PersonQualityMetrics:
    """Data class for person quality analysis results"""
    local_blur_score: float
    lighting_quality: float
    contrast_score: float
    overall_quality: float
    quality_level: PersonQualityLevel
    relative_sharpness: float  # Compared to image background


class PersonQualityAnalyzer:
    """
    Advanced quality analysis for the dominant person in an image
    Focuses on person-specific metrics rather than global image quality
    """
    
    def __init__(self):
        """Initialize the person quality analyzer"""
        self.blur_threshold = 100  # Laplacian variance threshold for sharpness
        self.contrast_threshold = 50  # Minimum acceptable contrast
        self.lighting_threshold = 0.3  # Minimum lighting quality score
        
        # Quality score weights
        self.weights = {
            'local_blur': 0.4,      # 40% - Most important for person quality
            'lighting': 0.3,        # 30% - Important for facial visibility
            'contrast': 0.2,        # 20% - Important for person separation from background
            'relative_sharpness': 0.1  # 10% - Bonus for person being sharper than background
        }
    
    def analyze_person_quality(self, person_bbox: Tuple[int, int, int, int], 
                             full_image: np.ndarray, 
                             dominant_person_data: Optional[Dict] = None) -> PersonQualityMetrics:
        """
        Comprehensive quality analysis of a person in the image
        
        Args:
            person_bbox: Bounding box of the person (x, y, w, h)
            full_image: Complete image in BGR format
            dominant_person_data: Additional data about the dominant person
            
        Returns:
            PersonQualityMetrics with detailed analysis results
        """
        try:
            if full_image is None:
                raise ValueError("Imagem não pode ser None")
            
            # Extract person ROI with padding
            person_roi = self._extract_person_roi(person_bbox, full_image, padding_factor=0.1)
            
            if person_roi is None:
                logger.warning("Não foi possível extrair ROI da pessoa")
                return self._get_default_quality_metrics()
            
            # Analyze local blur in person region
            local_blur_score = self._calculate_local_blur(person_roi)
            
            # Analyze lighting quality on the person
            lighting_quality = self._analyze_person_lighting(person_roi)
            
            # Calculate local contrast
            contrast_score = self._calculate_local_contrast(person_roi)
            
            # Calculate relative sharpness compared to background
            relative_sharpness = self._calculate_relative_sharpness(
                person_roi, full_image, person_bbox
            )
            
            # Combine all metrics into overall quality score
            overall_quality = self._combine_quality_metrics(
                local_blur_score, lighting_quality, contrast_score, relative_sharpness
            )
            
            # Determine quality level
            quality_level = self._classify_quality_level(overall_quality)
            
            return PersonQualityMetrics(
                local_blur_score=local_blur_score,
                lighting_quality=lighting_quality,
                contrast_score=contrast_score,
                overall_quality=overall_quality,
                quality_level=quality_level,
                relative_sharpness=relative_sharpness
            )
            
        except Exception as e:
            logger.error(f"Erro na análise de qualidade da pessoa: {e}")
            return self._get_default_quality_metrics()
    
    def _extract_person_roi(self, bbox: Tuple[int, int, int, int], 
                           image: np.ndarray, padding_factor: float = 0.1) -> Optional[np.ndarray]:
        """
        Extract person Region of Interest with optional padding
        """
        try:
            x, y, w, h = bbox
            img_height, img_width = image.shape[:2]
            
            # Add padding around the person
            padding_w = int(w * padding_factor)
            padding_h = int(h * padding_factor)
            
            # Calculate expanded boundaries
            x1 = max(0, x - padding_w)
            y1 = max(0, y - padding_h)
            x2 = min(img_width, x + w + padding_w)
            y2 = min(img_height, y + h + padding_h)
            
            # Extract ROI
            roi = image[y1:y2, x1:x2]
            
            if roi.size == 0:
                return None
                
            return roi
            
        except Exception as e:
            logger.error(f"Erro ao extrair ROI da pessoa: {e}")
            return None
    
    def _calculate_local_blur(self, roi: np.ndarray) -> float:
        """
        Calculate blur score specifically for the person region using Laplacian variance
        Higher values indicate sharper images
        """
        try:
            if len(roi.shape) == 3:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_roi = roi
            
            # Calculate Laplacian variance for blur detection
            laplacian_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
            
            # Normalize to 0-1 scale (higher = sharper)
            normalized_score = min(laplacian_var / 500.0, 1.0)  # 500 is empirical max for good quality
            
            return float(normalized_score)
            
        except Exception as e:
            logger.error(f"Erro ao calcular blur local: {e}")
            return 0.0
    
    def _analyze_person_lighting(self, roi: np.ndarray) -> float:
        """
        Analyze lighting quality specifically for the person
        Focuses on facial region and overall person illumination
        """
        try:
            # Convert to LAB color space for better lighting analysis
            lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            l_channel = lab_roi[:, :, 0]
            
            # Calculate lighting metrics
            mean_brightness = float(np.mean(l_channel))
            brightness_std = float(np.std(l_channel))
            
            # Check for proper exposure (not too dark, not too bright)
            exposure_score = self._evaluate_exposure_quality(mean_brightness)
            
            # Check for good contrast (standard deviation indicates detail)
            contrast_score = min(brightness_std / 50.0, 1.0)  # Normalize std dev
            
            # Detect shadows and highlights
            shadow_ratio = np.sum(l_channel < 50) / l_channel.size
            highlight_ratio = np.sum(l_channel > 200) / l_channel.size
            
            # Penalize excessive shadows or highlights
            shadow_penalty = min(shadow_ratio * 2, 0.3)  # Max 30% penalty
            highlight_penalty = min(highlight_ratio * 2, 0.3)  # Max 30% penalty
            
            # Combined lighting score
            lighting_score = (exposure_score * 0.5 + contrast_score * 0.5) - shadow_penalty - highlight_penalty
            lighting_score = max(0.0, min(1.0, lighting_score))
            
            return float(lighting_score)
            
        except Exception as e:
            logger.error(f"Erro ao analisar iluminação da pessoa: {e}")
            return 0.5
    
    def _evaluate_exposure_quality(self, mean_brightness: float) -> float:
        """
        Evaluate exposure quality based on mean brightness
        Optimal range is around 100-150 in LAB L channel
        """
        optimal_min, optimal_max = 80, 170
        
        if optimal_min <= mean_brightness <= optimal_max:
            # Within optimal range
            return 1.0
        elif mean_brightness < optimal_min:
            # Too dark - gradual penalty
            return max(0.0, mean_brightness / optimal_min)
        else:
            # Too bright - gradual penalty
            return max(0.0, (255 - mean_brightness) / (255 - optimal_max))
    
    def _calculate_local_contrast(self, roi: np.ndarray) -> float:
        """
        Calculate local contrast in the person region
        Higher contrast usually indicates better definition
        """
        try:
            if len(roi.shape) == 3:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_roi = roi
            
            # Method 1: Standard deviation of pixel intensities
            std_contrast = np.std(gray_roi)
            
            # Method 2: Michelson contrast (max-min)/(max+min)
            min_val, max_val = np.min(gray_roi), np.max(gray_roi)
            if max_val + min_val > 0:
                michelson_contrast = (max_val - min_val) / (max_val + min_val)
            else:
                michelson_contrast = 0.0
            
            # Combine both methods
            combined_contrast = (std_contrast / 127.5) * 0.7 + michelson_contrast * 0.3
            combined_contrast = min(1.0, combined_contrast)
            
            return float(combined_contrast)
            
        except Exception as e:
            logger.error(f"Erro ao calcular contraste local: {e}")
            return 0.5
    
    def _calculate_relative_sharpness(self, person_roi: np.ndarray, 
                                    full_image: np.ndarray, 
                                    person_bbox: Tuple[int, int, int, int]) -> float:
        """
        Calculate how sharp the person is relative to the background
        Positive values indicate person is sharper than background
        """
        try:
            # Calculate person sharpness
            person_sharpness = self._calculate_local_blur(person_roi)
            
            # Create background mask (exclude person area)
            background_roi = self._extract_background_sample(full_image, person_bbox)
            background_sharpness = self._calculate_local_blur(background_roi) if background_roi is not None else 0.5
            
            # Calculate relative sharpness (-1 to 1 scale)
            if background_sharpness > 0:
                relative_sharpness = (person_sharpness - background_sharpness) / max(person_sharpness, background_sharpness)
            else:
                relative_sharpness = 0.0
            
            # Normalize to 0-1 scale (0.5 = equal sharpness, 1.0 = person much sharper)
            normalized_relative = (relative_sharpness + 1) / 2
            
            return float(normalized_relative)
            
        except Exception as e:
            logger.error(f"Erro ao calcular nitidez relativa: {e}")
            return 0.5
    
    def _extract_background_sample(self, image: np.ndarray, 
                                 person_bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract a sample of the background (areas not occupied by the person)
        """
        try:
            x, y, w, h = person_bbox
            img_height, img_width = image.shape[:2]
            
            # Create mask for person area
            mask = np.ones((img_height, img_width), dtype=np.uint8) * 255
            mask[y:y+h, x:x+w] = 0  # Exclude person area
            
            # Extract background pixels
            background_pixels = image[mask == 255]
            
            if len(background_pixels) > 1000:  # Ensure we have enough pixels for analysis
                # Reshape to image-like format for blur analysis
                bg_sample_size = min(len(background_pixels), 10000)  # Limit for performance
                background_sample = background_pixels[:bg_sample_size].reshape(-1, 1, 3)
                return background_sample
            
            return None
            
        except Exception as e:
            logger.error(f"Erro ao extrair amostra do fundo: {e}")
            return None
    
    def _combine_quality_metrics(self, local_blur: float, lighting: float, 
                               contrast: float, relative_sharpness: float) -> float:
        """
        Combine all quality metrics into a single overall score
        """
        overall_score = (
            local_blur * self.weights['local_blur'] +
            lighting * self.weights['lighting'] +
            contrast * self.weights['contrast'] +
            relative_sharpness * self.weights['relative_sharpness']
        )
        
        return max(0.0, min(1.0, overall_score))
    
    def _classify_quality_level(self, overall_score: float) -> PersonQualityLevel:
        """
        Classify the overall quality score into discrete levels
        """
        if overall_score >= 0.8:
            return PersonQualityLevel.EXCELLENT
        elif overall_score >= 0.6:
            return PersonQualityLevel.GOOD
        elif overall_score >= 0.4:
            return PersonQualityLevel.ACCEPTABLE
        else:
            return PersonQualityLevel.POOR
    
    def _get_default_quality_metrics(self) -> PersonQualityMetrics:
        """
        Return default quality metrics in case of analysis failure
        """
        return PersonQualityMetrics(
            local_blur_score=0.0,
            lighting_quality=0.0,
            contrast_score=0.0,
            overall_quality=0.0,
            quality_level=PersonQualityLevel.POOR,
            relative_sharpness=0.0
        )


def analyze_person_quality_batch(image_paths: List[str], 
                               person_bboxes: List[Tuple[int, int, int, int]]) -> List[PersonQualityMetrics]:
    """
    Batch analysis of person quality for multiple images
    
    Args:
        image_paths: List of paths to images
        person_bboxes: List of person bounding boxes corresponding to each image
        
    Returns:
        List of PersonQualityMetrics for each image
    """
    analyzer = PersonQualityAnalyzer()
    results = []
    
    for image_path, bbox in zip(image_paths, person_bboxes):
        try:
            image = cv2.imread(image_path)
            if image is not None:
                quality_metrics = analyzer.analyze_person_quality(bbox, image)
                results.append(quality_metrics)
            else:
                logger.warning(f"Não foi possível carregar imagem: {image_path}")
                results.append(analyzer._get_default_quality_metrics())
        except Exception as e:
            logger.error(f"Erro ao processar {image_path}: {e}")
            results.append(analyzer._get_default_quality_metrics())
    
    return results


if __name__ == "__main__":
    # Example usage and testing
    import sys
    import os
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with a sample image
    test_image_path = "data/input/IMG_0001.JPG"
    if os.path.exists(test_image_path):
        print("🔍 Testando PersonQualityAnalyzer...")
        
        # Load test image
        test_image = cv2.imread(test_image_path)
        
        # Mock person bounding box (would come from person detector)
        # This is just for testing - normally comes from PersonDetector
        test_bbox = (800, 400, 600, 1000)  # x, y, w, h
        
        # Analyze person quality
        analyzer = PersonQualityAnalyzer()
        quality_metrics = analyzer.analyze_person_quality(test_bbox, test_image)
        
        print(f"   ✅ Blur local: {quality_metrics.local_blur_score:.3f}")
        print(f"   ✅ Qualidade de iluminação: {quality_metrics.lighting_quality:.3f}")
        print(f"   ✅ Contraste local: {quality_metrics.contrast_score:.3f}")
        print(f"   ✅ Nitidez relativa: {quality_metrics.relative_sharpness:.3f}")
        print(f"   ✅ Qualidade geral: {quality_metrics.overall_quality:.3f}")
        print(f"   ✅ Nível de qualidade: {quality_metrics.quality_level.value}")
        
        print("🎉 PersonQualityAnalyzer funcionando!")
    else:
        print(f"❌ Imagem de teste não encontrada: {test_image_path}")
