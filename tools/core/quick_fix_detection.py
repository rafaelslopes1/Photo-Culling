#!/usr/bin/env python3
"""
Quick Fix Detection System - Sistema de Correção Rápida
Corrige os problemas principais: detecção de pessoas e métricas corretas
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.person_detector import PersonDetector
from src.core.face_recognition_system import FaceRecognitionSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QuickFixDetectionSystem:
    """
    Sistema de correção rápida dos problemas principais
    """
    
    def __init__(self):
        self.person_detector = None
        self.face_recognition_system = None
        
    def initialize(self):
        """Initialize detection systems"""
        try:
            self.person_detector = PersonDetector()
            self.face_recognition_system = FaceRecognitionSystem()
            logger.info("✅ Sistemas inicializados")
            return True
        except Exception as e:
            logger.error(f"❌ Erro na inicialização: {e}")
            return False
    
    def force_person_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Force person detection using MediaPipe Pose directly
        """
        detected_persons = []
        
        try:
            import mediapipe as mp
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width = image.shape[:2]
            
            logger.info("   🔍 Forçando detecção de pose MediaPipe...")
            
            # Initialize pose detector with lower confidence
            mp_pose = mp.solutions.pose
            pose_detector = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.1,  # Very low threshold
                min_tracking_confidence=0.1
            )
            
            pose_results = pose_detector.process(rgb_image)
            
            if pose_results.pose_landmarks:
                # Extract landmarks
                landmarks = []
                for landmark in pose_results.pose_landmarks.landmark:
                    x = int(landmark.x * image_width)
                    y = int(landmark.y * image_height)
                    visibility = landmark.visibility
                    landmarks.append({'x': x, 'y': y, 'visibility': visibility})
                
                # Calculate bounding box from landmarks
                valid_landmarks = [l for l in landmarks if l['visibility'] > 0.1]
                
                if valid_landmarks:
                    x_coords = [l['x'] for l in valid_landmarks]
                    y_coords = [l['y'] for l in valid_landmarks]
                    
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # Add padding (20% of bbox size)
                    width = x_max - x_min
                    height = y_max - y_min
                    padding_x = int(width * 0.2)
                    padding_y = int(height * 0.2)
                    
                    x = max(0, x_min - padding_x)
                    y = max(0, y_min - padding_y)
                    w = min(image_width - x, x_max - x + padding_x)
                    h = min(image_height - y, y_max - y + padding_y)
                    
                    detected_persons.append({
                        'source': 'mediapipe_pose_forced',
                        'bbox': [x, y, w, h],
                        'confidence': 0.8,  # High confidence since we found pose
                        'pose_landmarks': landmarks,
                        'landmark_count': len(valid_landmarks)
                    })
                    
                    logger.info(f"      ✅ Pessoa detectada via pose forçada ({len(valid_landmarks)} landmarks)")
            else:
                logger.info("      ❌ Nenhuma pose detectada mesmo com threshold baixo")
            
            pose_detector.close()
            
        except Exception as e:
            logger.error(f"❌ Erro na detecção forçada de pessoas: {e}")
        
        return detected_persons
    
    def calculate_correct_metrics(self, image: np.ndarray, faces: List[Dict], persons: List[Dict]) -> Dict:
        """
        Calculate correct quality metrics
        """
        metrics = {}
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Blur detection (Laplacian variance)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            metrics['blur_score'] = float(blur_score)
            
            # Brightness
            brightness = np.mean(gray)
            metrics['brightness'] = float(brightness)
            
            # Correct face and person counts
            metrics['face_count'] = len(faces)
            metrics['person_count'] = len(persons)
            
            # Exposure analysis
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # Check for overexposure (bright pixels)
            overexposed_pixels = np.sum(hist[240:])
            total_pixels = gray.shape[0] * gray.shape[1]
            overexposure_ratio = overexposed_pixels / total_pixels
            
            # Check for underexposure (dark pixels)
            underexposed_pixels = np.sum(hist[0:16])
            underexposure_ratio = underexposed_pixels / total_pixels
            
            metrics['overexposure_ratio'] = float(overexposure_ratio)
            metrics['underexposure_ratio'] = float(underexposure_ratio)
            
            # Quality assessment
            if blur_score < 50:
                quality_level = "blurry"
                recommendation = "reject"
            elif blur_score < 100:
                quality_level = "poor"
                recommendation = "analyze_manually"
            elif blur_score < 150:
                quality_level = "fair"
                recommendation = "analyze_manually"
            else:
                quality_level = "good"
                recommendation = "keep"
            
            # Consider exposure in recommendation
            if overexposure_ratio > 0.1 or underexposure_ratio > 0.3:
                if recommendation == "keep":
                    recommendation = "analyze_manually"
                quality_level += "_exposure_issues"
            
            metrics['quality_level'] = quality_level
            metrics['recommendation'] = recommendation
            metrics['quality_score'] = min(1.0, blur_score / 200.0)  # Normalized 0-1
            
        except Exception as e:
            logger.error(f"❌ Erro no cálculo de métricas: {e}")
            metrics = {
                'blur_score': 0,
                'brightness': 0,
                'face_count': len(faces),
                'person_count': len(persons),
                'quality_level': 'unknown',
                'recommendation': 'analyze_manually',
                'quality_score': 0,
                'overexposure_ratio': 0,
                'underexposure_ratio': 0
            }
        
        return metrics
    
    def test_quick_fix(self, image_path: str):
        """
        Test quick fixes on a single image
        """
        logger.info(f"🔧 TESTE DE CORREÇÃO RÁPIDA: {os.path.basename(image_path)}")
        logger.info("=" * 60)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error("❌ Não foi possível carregar a imagem")
            return
        
        # 1. Standard detection first
        logger.info("📊 DETECÇÃO PADRÃO:")
        try:
            standard_result = self.person_detector.detect_persons_and_faces(image)
            standard_faces = standard_result.get('faces', [])
            standard_persons_raw = standard_result.get('persons', [])
            
            # Convert PersonDetection objects to dicts
            standard_persons = []
            for person in standard_persons_raw:
                person_dict = {
                    'source': 'face_based_detection',
                    'bbox': person.bounding_box,
                    'confidence': person.confidence,
                    'dominance_score': person.dominance_score,
                    'area_ratio': person.area_ratio,
                    'centrality': person.centrality,
                    'local_sharpness': person.local_sharpness,
                    'pose_landmarks': person.pose_landmarks or []
                }
                standard_persons.append(person_dict)
            
            logger.info(f"   Faces padrão: {len(standard_faces)}")
            logger.info(f"   Pessoas padrão: {len(standard_persons)}")
        except Exception as e:
            logger.error(f"   ❌ Erro na detecção padrão: {e}")
            standard_faces = []
            standard_persons = []
        
        # 2. Force person detection
        logger.info("🎯 DETECÇÃO FORÇADA DE PESSOAS:")
        forced_persons = self.force_person_detection(image)
        
        # 3. Combine results
        all_persons = standard_persons + forced_persons
        all_faces = standard_faces
        
        logger.info(f"   Total pessoas (incluindo forçada): {len(all_persons)}")
        logger.info(f"   Total rostos: {len(all_faces)}")
        
        # 4. Calculate correct metrics
        logger.info("📊 MÉTRICAS CORRIGIDAS:")
        metrics = self.calculate_correct_metrics(image, all_faces, all_persons)
        
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"   {key}: {value:.3f}")
            else:
                logger.info(f"   {key}: {value}")
        
        # 5. Create corrected annotated image
        annotated = image.copy()
        
        # Draw all faces
        for i, face in enumerate(all_faces):
            bbox = face.get('bbox', [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                confidence = face.get('confidence', 0)
                label = f"Face {i+1} ({confidence:.2f})"
                cv2.putText(annotated, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw all persons
        for i, person in enumerate(all_persons):
            bbox = person.get('bbox', [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                color = (0, 255, 255) if person['source'].startswith('mediapipe_pose') else (0, 255, 0)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
                
                source = person.get('source', 'unknown')
                confidence = person.get('confidence', 0)
                label = f"Person {i+1} ({source[:8]}) {confidence:.2f}"
                cv2.putText(annotated, label, (x, y - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw pose landmarks if available
                pose_landmarks = person.get('pose_landmarks', [])
                for landmark in pose_landmarks:
                    if isinstance(landmark, dict) and 'x' in landmark and 'y' in landmark:
                        if landmark.get('visibility', 0) > 0.5:
                            cv2.circle(annotated, (landmark['x'], landmark['y']), 2, (255, 255, 0), -1)
        
        # Add corrected metrics overlay
        y_offset = 30
        metric_texts = [
            f"Faces: {metrics['face_count']}",
            f"Persons: {metrics['person_count']}",
            f"Blur: {metrics['blur_score']:.1f}",
            f"Brightness: {metrics['brightness']:.1f}",
            f"Quality: {metrics['quality_level']}",
            f"Rec: {metrics['recommendation']}",
            f"Overexp: {metrics['overexposure_ratio']:.3f}",
            f"Underexp: {metrics['underexposure_ratio']:.3f}"
        ]
        
        for i, text in enumerate(metric_texts):
            y_pos = y_offset + (i * 25)
            cv2.rectangle(annotated, (10, y_pos - 15), (250, y_pos + 5), (0, 0, 0), -1)
            cv2.putText(annotated, text, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Save result
        output_dir = Path("data/analysis_results/quick_fix")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{Path(image_path).stem}_quick_fix.jpg"
        cv2.imwrite(str(output_path), annotated)
        
        # Save JSON with detailed results
        json_path = output_dir / f"{Path(image_path).stem}_quick_fix.json"
        result_data = {
            'image_path': image_path,
            'standard_detection': {
                'faces': len(standard_faces),
                'persons': len(standard_persons)
            },
            'enhanced_detection': {
                'faces': len(all_faces),
                'persons': len(all_persons),
                'forced_persons': len(forced_persons)
            },
            'metrics': metrics,
            'person_detection_sources': [p.get('source', 'unknown') for p in all_persons]
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Resultado salvo: {output_path}")
        logger.info(f"✅ JSON salvo: {json_path}")
        logger.info("=" * 60)
        
        return result_data


def main():
    """Test quick fixes"""
    system = QuickFixDetectionSystem()
    
    if not system.initialize():
        return
    
    # Test with problematic images
    test_images = [
        "data/input/IMG_0252.JPG",  # 5 faces but only 3 detected + people
        "data/input/IMG_0001.JPG",  # Single person
        "data/input/IMG_0304.JPG",  # Multiple faces
        "data/input/IMG_0339.JPG",  # People detection issue
        "data/input/IMG_0400.JPG"   # Another test case
    ]
    
    results = []
    
    for image_path in test_images:
        if os.path.exists(image_path):
            try:
                result = system.test_quick_fix(image_path)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"❌ Erro ao processar {image_path}: {e}")
        else:
            logger.warning(f"⚠️ Imagem não encontrada: {image_path}")
    
    # Generate summary
    if results:
        summary = {
            'total_images': len(results),
            'average_faces': sum(r['enhanced_detection']['faces'] for r in results) / len(results),
            'average_persons': sum(r['enhanced_detection']['persons'] for r in results) / len(results),
            'forced_detection_success': sum(1 for r in results if r['enhanced_detection']['forced_persons'] > 0),
            'images_with_people': sum(1 for r in results if r['enhanced_detection']['persons'] > 0),
            'quality_distribution': {}
        }
        
        # Quality distribution
        for result in results:
            quality = result['metrics']['quality_level']
            summary['quality_distribution'][quality] = summary['quality_distribution'].get(quality, 0) + 1
        
        summary_path = "data/analysis_results/quick_fix/quick_fix_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 RESUMO FINAL salvo em: {summary_path}")
        logger.info(f"   Total de imagens: {summary['total_images']}")
        logger.info(f"   Faces por imagem (média): {summary['average_faces']:.1f}")
        logger.info(f"   Pessoas por imagem (média): {summary['average_persons']:.1f}")
        logger.info(f"   Detecção forçada bem-sucedida: {summary['forced_detection_success']}/{len(results)}")


if __name__ == "__main__":
    main()
