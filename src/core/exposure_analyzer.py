#!/usr/bin/env python3
"""
Exposure Analysis Module for Photo Culling System
Módulo de análise de exposição para sistema de seleção de fotos

Implements exposure analysis using HSV histogram and adaptive thresholding
Following the roadmap Phase 1 specifications
"""

import cv2
import numpy as np
from enum import Enum
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ExposureLevel(Enum):
    """Classification levels for image exposure"""
    EXTREMELY_DARK = "extremely_dark"
    DARK = "dark"
    ADEQUATE = "adequate"
    BRIGHT = "bright"
    EXTREMELY_BRIGHT = "extremely_bright"


class ExposureAnalyzer:
    """
    Advanced exposure analysis using HSV histogram and adaptive thresholding
    Based on scientific methods from computer vision literature
    """
    
    def __init__(self):
        """Initialize the exposure analyzer with default settings"""
        self.thresholds = {
            'extremely_dark': 40,
            'dark': 80,
            'bright': 180,
            'extremely_bright': 220
        }
    
    def analyze_exposure(self, image: np.ndarray) -> Dict:
        """
        Analyze image exposure using HSV histogram and adaptive thresholding
        
        Args:
            image: Input image in BGR format (OpenCV format)
            
        Returns:
            Dictionary containing exposure analysis results
        """
        try:
            if image is None:
                raise ValueError("Imagem não pode ser None")
            
            # Convert to HSV for better brightness analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            value_channel = hsv[:, :, 2]
            
            # Calculate histogram for the Value channel
            hist = cv2.calcHist([value_channel], [0], None, [256], [0, 256])
            
            # Otsu threshold for adaptive analysis
            try:
                threshold, _ = cv2.threshold(value_channel, 0, 255, 
                                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            except Exception as e:
                logger.warning(f"Erro ao calcular threshold de Otsu: {e}")
                threshold = 128  # Default fallback
            
            # Calculate mean brightness
            mean_brightness = float(np.mean(value_channel))
            
            # Calculate histogram statistics
            hist_stats = self._calculate_histogram_statistics(hist, value_channel)
            
            # Determine exposure classification
            exposure_level = self._classify_exposure(mean_brightness)
            
            # Calculate exposure quality score (0-1, where 1 is best)
            quality_score = self._calculate_exposure_quality(mean_brightness, hist_stats)
            
            return {
                'exposure_level': exposure_level.value,
                'mean_brightness': float(mean_brightness),
                'otsu_threshold': float(threshold),
                'quality_score': float(quality_score),
                'histogram_stats': hist_stats,
                'is_properly_exposed': quality_score > 0.6,
                'analysis_version': '1.0'
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de exposição: {e}")
            return self._get_default_exposure_result()
    
    def _calculate_histogram_statistics(self, hist: np.ndarray, 
                                       value_channel: np.ndarray) -> Dict:
        """Calculate comprehensive histogram statistics"""
        try:
            # Normalize histogram
            hist_norm = hist.flatten() / hist.sum()
            
            # Calculate percentiles
            percentiles = np.percentile(value_channel, [5, 25, 50, 75, 95])
            
            # Calculate histogram moments
            bins = np.arange(256)
            mean_hist = np.sum(bins * hist_norm)
            variance_hist = np.sum((bins - mean_hist) ** 2 * hist_norm)
            std_hist = np.sqrt(variance_hist)
            
            # Detect clipping (overexposure/underexposure)
            shadow_clipping = np.sum(hist_norm[:10])  # First 10 bins
            highlight_clipping = np.sum(hist_norm[-10:])  # Last 10 bins
            
            return {
                'percentile_5': float(percentiles[0]),
                'percentile_25': float(percentiles[1]),
                'median': float(percentiles[2]),
                'percentile_75': float(percentiles[3]),
                'percentile_95': float(percentiles[4]),
                'mean_histogram': float(mean_hist),
                'std_histogram': float(std_hist),
                'shadow_clipping': float(shadow_clipping),
                'highlight_clipping': float(highlight_clipping),
                'dynamic_range': float(percentiles[4] - percentiles[0])
            }
            
        except Exception as e:
            logger.error(f"Erro ao calcular estatísticas do histograma: {e}")
            return {}
    
    def _classify_exposure(self, mean_brightness: float) -> ExposureLevel:
        """Classify exposure level based on mean brightness"""
        if mean_brightness < self.thresholds['extremely_dark']:
            return ExposureLevel.EXTREMELY_DARK
        elif mean_brightness < self.thresholds['dark']:
            return ExposureLevel.DARK
        elif mean_brightness > self.thresholds['extremely_bright']:
            return ExposureLevel.EXTREMELY_BRIGHT
        elif mean_brightness > self.thresholds['bright']:
            return ExposureLevel.BRIGHT
        else:
            return ExposureLevel.ADEQUATE
    
    def _calculate_exposure_quality(self, mean_brightness: float, 
                                   hist_stats: Dict) -> float:
        """
        Calculate exposure quality score (0-1)
        Higher score means better exposure
        """
        try:
            # Base score from brightness (optimal around 100-150)
            optimal_brightness = 125
            brightness_diff = abs(mean_brightness - optimal_brightness)
            brightness_score = max(0, 1 - (brightness_diff / 100))
            
            # Penalty for clipping
            clipping_penalty = 0
            if hist_stats:
                shadow_clip = hist_stats.get('shadow_clipping', 0)
                highlight_clip = hist_stats.get('highlight_clipping', 0)
                clipping_penalty = min(0.5, (shadow_clip + highlight_clip) * 2)
            
            # Dynamic range bonus
            dynamic_range_bonus = 0
            if hist_stats and 'dynamic_range' in hist_stats:
                dynamic_range = hist_stats['dynamic_range']
                # Good dynamic range is typically > 150
                if dynamic_range > 150:
                    dynamic_range_bonus = 0.1
            
            # Calculate final score
            final_score = brightness_score - clipping_penalty + dynamic_range_bonus
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logger.error(f"Erro ao calcular qualidade da exposição: {e}")
            return 0.5  # Default neutral score
    
    def _get_default_exposure_result(self) -> Dict:
        """Return default result in case of analysis failure"""
        return {
            'exposure_level': ExposureLevel.ADEQUATE.value,
            'mean_brightness': 128.0,
            'otsu_threshold': 128.0,
            'quality_score': 0.5,
            'histogram_stats': {},
            'is_properly_exposed': True,
            'analysis_version': '1.0',
            'error': True
        }


def analyze_image_exposure(image_path: str) -> Optional[Dict]:
    """
    Convenience function to analyze exposure of an image file
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Exposure analysis results or None if failed
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Não foi possível carregar a imagem: {image_path}")
            return None
        
        analyzer = ExposureAnalyzer()
        return analyzer.analyze_exposure(image)
        
    except Exception as e:
        logger.error(f"Erro ao analisar exposição da imagem {image_path}: {e}")
        return None


if __name__ == "__main__":
    # Test the module
    import sys
    if len(sys.argv) > 1:
        result = analyze_image_exposure(sys.argv[1])
        if result:
            print("Exposure Analysis Results:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        else:
            print("Failed to analyze image")
