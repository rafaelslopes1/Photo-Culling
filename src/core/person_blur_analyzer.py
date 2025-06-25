#!/usr/bin/env python3
"""
Person-Focused Blur Analyzer - Analisador de Blur Focado na Pessoa
Implementa análise de nitidez específica para regiões de pessoas detectadas
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PersonBlurAnalyzer:
    """
    Analisador de blur focado especificamente nas pessoas detectadas
    """
    
    def __init__(self):
        # Weights for different regions
        self.face_weight = 0.6  # Face blur is most important
        self.body_weight = 0.4  # Body blur is secondary
        
        # Blur thresholds specific to persons
        self.person_blur_thresholds = {
            'excellent': 80,    # Person is very sharp
            'good': 50,         # Person is adequately sharp  
            'fair': 30,         # Person has some blur but usable
            'poor': 15,         # Person is significantly blurred
            'reject': 0         # Person is too blurred
        }
    
    def extract_person_region(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Extract person region from image using bounding box
        
        Args:
            image: Input image
            bbox: [x, y, width, height] bounding box
            
        Returns:
            Cropped person region
        """
        try:
            x, y, w, h = bbox
            
            # Ensure coordinates are within image bounds
            height, width = image.shape[:2]
            x = max(0, min(x, width))
            y = max(0, min(y, height))
            w = max(1, min(w, width - x))
            h = max(1, min(h, height - y))
            
            return image[y:y+h, x:x+w]
            
        except Exception as e:
            logger.error(f"Erro ao extrair região da pessoa: {e}")
            return image
    
    def extract_face_region(self, person_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face region from person image using face detection
        
        Args:
            person_image: Cropped person image
            
        Returns:
            Face region if detected, None otherwise
        """
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)
            
            # Use Haar cascade for face detection
            try:
                # Try different possible paths for face cascade
                import os
                possible_paths = [
                    'haarcascade_frontalface_default.xml',
                    '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                    '/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
                ]
                
                face_cascade = None
                for path in possible_paths:
                    if os.path.exists(path):
                        face_cascade = cv2.CascadeClassifier(path)
                        break
                
                if face_cascade is None:
                    face_cascade = cv2.CascadeClassifier()
                    
            except Exception as e:
                logger.warning(f"Falha ao carregar classificador de face: {e}")
                return None
            
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Use the largest face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest_face
                
                # Add some padding around face
                padding = int(min(w, h) * 0.1)
                x = max(0, x - padding)
                y = max(0, y - padding) 
                w = min(person_image.shape[1] - x, w + 2*padding)
                h = min(person_image.shape[0] - y, h + 2*padding)
                
                return person_image[y:y+h, x:x+w]
            
            return None
            
        except Exception as e:
            logger.error(f"Erro na detecção de face: {e}")
            return None
    
    def calculate_region_blur(self, region: np.ndarray) -> float:
        """
        Calculate blur score for a specific region using multiple methods
        
        Args:
            region: Image region to analyze
            
        Returns:
            Blur score (higher = sharper)
        """
        try:
            if region is None or region.size == 0:
                return 0.0
            
            # Convert to grayscale
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region
            
            # Method 1: Variance of Laplacian (primary)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Method 2: Sobel gradient magnitude (secondary)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2).mean()
            
            # Combine methods (Laplacian is more reliable for blur detection)
            combined_score = laplacian_var * 0.8 + sobel_magnitude * 0.2
            
            return float(combined_score)
            
        except Exception as e:
            logger.error(f"Erro no cálculo de blur: {e}")
            return 0.0
    
    def analyze_person_blur(self, image: np.ndarray, person_bbox: List[int], 
                           face_bbox: Optional[List[int]] = None) -> Dict:
        """
        Analyze blur specifically for a detected person
        
        Args:
            image: Full image
            person_bbox: Person bounding box [x, y, w, h]
            face_bbox: Optional face bounding box within person region
            
        Returns:
            Dictionary with person blur analysis
        """
        try:
            # Extract person region
            person_region = self.extract_person_region(image, person_bbox)
            
            if person_region.size == 0:
                return self._default_blur_analysis()
            
            # Calculate body blur (full person region)
            body_blur = self.calculate_region_blur(person_region)
            
            # Calculate face blur if face is available
            face_blur = None
            if face_bbox:
                # Face bbox is relative to full image, adjust for person region
                face_region = self.extract_person_region(image, face_bbox)
                face_blur = self.calculate_region_blur(face_region)
            else:
                # Try to detect face within person region
                face_region = self.extract_face_region(person_region)
                if face_region is not None:
                    face_blur = self.calculate_region_blur(face_region)
            
            # Combine scores
            if face_blur is not None:
                # Weighted combination: face is more important
                combined_blur = (face_blur * self.face_weight + 
                               body_blur * self.body_weight)
                blur_focus = "face_and_body"
            else:
                # Only body blur available
                combined_blur = body_blur
                blur_focus = "body_only"
            
            # Determine quality level
            quality_level = self._determine_person_blur_quality(combined_blur)
            
            return {
                'person_blur_score': float(combined_blur),
                'face_blur_score': float(face_blur) if face_blur else None,
                'body_blur_score': float(body_blur),
                'blur_focus': blur_focus,
                'quality_level': quality_level,
                'is_person_sharp': combined_blur >= self.person_blur_thresholds['good'],
                'recommended_use': self._get_usage_recommendation(quality_level),
                'analysis_method': 'person_focused_blur_v1.0'
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de blur da pessoa: {e}")
            return self._default_blur_analysis()
    
    def _determine_person_blur_quality(self, blur_score: float) -> str:
        """
        Determine quality level based on person blur score
        """
        if blur_score >= self.person_blur_thresholds['excellent']:
            return 'excellent'
        elif blur_score >= self.person_blur_thresholds['good']:
            return 'good'
        elif blur_score >= self.person_blur_thresholds['fair']:
            return 'fair'
        elif blur_score >= self.person_blur_thresholds['poor']:
            return 'poor'
        else:
            return 'reject'
    
    def _get_usage_recommendation(self, quality_level: str) -> str:
        """
        Get usage recommendation based on quality level
        """
        recommendations = {
            'excellent': 'Ideal para impressão e uso profissional',
            'good': 'Adequada para a maioria dos usos',
            'fair': 'Utilizável com pequenos ajustes ou para web',
            'poor': 'Uso limitado, considerar descarte',
            'reject': 'Pessoa muito desfocada, recomendar descarte'
        }
        return recommendations.get(quality_level, 'Análise inconclusiva')
    
    def _default_blur_analysis(self) -> Dict:
        """
        Return default analysis when calculation fails
        """
        return {
            'person_blur_score': 0.0,
            'face_blur_score': None,
            'body_blur_score': 0.0,
            'blur_focus': 'failed',
            'quality_level': 'unknown',
            'is_person_sharp': False,
            'recommended_use': 'Análise falhou',
            'analysis_method': 'person_focused_blur_v1.0'
        }
    
    def analyze_multiple_persons(self, image: np.ndarray, 
                                persons_data: List[Dict]) -> List[Dict]:
        """
        Analyze blur for multiple detected persons
        
        Args:
            image: Full image
            persons_data: List of person detection data
            
        Returns:
            List of blur analyses for each person
        """
        results = []
        
        for i, person in enumerate(persons_data):
            try:
                # Extract bbox from person data
                bbox = person.get('bbox', person.get('bounding_box'))
                face_bbox = person.get('face_bbox')
                
                if bbox:
                    analysis = self.analyze_person_blur(image, bbox, face_bbox)
                    analysis['person_index'] = i
                    analysis['person_confidence'] = person.get('confidence', 1.0)
                    results.append(analysis)
                
            except Exception as e:
                logger.error(f"Erro na análise da pessoa {i}: {e}")
                continue
        
        return results
    
    def get_dominant_person_blur(self, blur_analyses: List[Dict]) -> Optional[Dict]:
        """
        Get blur analysis for the dominant (highest confidence/largest) person
        
        Args:
            blur_analyses: List of person blur analyses
            
        Returns:
            Blur analysis for dominant person
        """
        if not blur_analyses:
            return None
        
        # Find person with highest confidence or largest area
        dominant = max(blur_analyses, 
                      key=lambda p: p.get('person_confidence', 0))
        
        return dominant


def main():
    """
    Test the person blur analyzer
    """
    print("🔍 ANALISADOR DE BLUR FOCADO NA PESSOA")
    print("=" * 50)
    
    analyzer = PersonBlurAnalyzer()
    
    # Test with sample data
    test_image_path = "data/input/TSL2- IMG  (336).JPG"
    
    try:
        import cv2
        image = cv2.imread(test_image_path)
        
        if image is not None:
            # Mock person detection data
            person_bbox = [685, 787, 477, 1528]  # From JSON data
            
            analysis = analyzer.analyze_person_blur(image, person_bbox)
            
            print(f"📸 Teste com: {test_image_path}")
            print(f"🔍 Blur da pessoa: {analysis['person_blur_score']:.1f}")
            print(f"😊 Blur do rosto: {analysis['face_blur_score']}")
            print(f"👤 Blur do corpo: {analysis['body_blur_score']:.1f}")
            print(f"⭐ Qualidade: {analysis['quality_level']}")
            print(f"💡 Recomendação: {analysis['recommended_use']}")
        else:
            print(f"❌ Não foi possível carregar: {test_image_path}")
            
    except Exception as e:
        print(f"❌ Erro no teste: {e}")


if __name__ == "__main__":
    main()
