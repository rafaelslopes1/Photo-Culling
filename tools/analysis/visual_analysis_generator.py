#!/usr/bin/env python3
"""
Visual Analysis Generator - Gerador de Análise Visual
Cria visualizações das imagens com todas as detecções e landmarks para análise visual
"""

import os
import sys
import cv2
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.feature_extractor import FeatureExtractor
from src.core.person_detector import PersonDetector
from src.core.face_recognition_system import FaceRecognitionSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VisualAnalysisGenerator:
    """
    Gera visualizações das imagens com todas as detecções para análise visual
    """
    
    def __init__(self):
        self.feature_extractor = None
        self.person_detector = None
        self.face_recognition_system = None
        self.output_dir = Path("data/analysis_results/visual_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def initialize_systems(self):
        """
        Initialize all detection systems
        """
        logger.info("🔄 Inicializando sistemas de detecção...")
        
        try:
            self.feature_extractor = FeatureExtractor()
            self.person_detector = PersonDetector()
            self.face_recognition_system = FaceRecognitionSystem()
            
            logger.info("✅ Sistemas inicializados com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro na inicialização: {e}")
            return False
    
    def analyze_image_with_visualizations(self, image_path: str) -> dict:
        """
        Analyze image and create visualizations with all detections
        """
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"❌ Não foi possível carregar: {image_path}")
            return {}
        
        filename = os.path.basename(image_path)
        logger.info(f"📸 Analisando: {filename}")
        
        # Create a copy for annotations
        annotated_image = image.copy()
        
        analysis_data = {
            'filename': filename,
            'image_shape': image.shape,
            'detections': {},
            'features': {},
            'visualizations_created': []
        }
        
        try:
            # 1. Person Detection with MediaPipe
            logger.info("   🔍 Detecção de pessoas...")
            person_results = self.person_detector.detect_persons_and_faces(image)
            
            if person_results and person_results.get('success'):
                persons = person_results.get('persons', [])
                analysis_data['detections']['persons'] = []
                
                for i, person in enumerate(persons):
                    person_data = {
                        'person_id': i,
                        'bbox': person.bounding_box,
                        'dominance_score': person.dominance_score,
                        'landmarks_count': len(person.landmarks) if person.landmarks else 0,
                        'pose_landmarks_count': len(person.pose_landmarks) if person.pose_landmarks else 0
                    }
                    analysis_data['detections']['persons'].append(person_data)
                    
                    # Draw person bounding box
                    x, y, w, h = person.bounding_box
                    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    
                    # Add label
                    label = f"Person {i} (Score: {person.dominance_score:.2f})"
                    cv2.putText(annotated_image, label, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw pose landmarks if available
                    if person.pose_landmarks:
                        for landmark in person.pose_landmarks:
                            if len(landmark) >= 2:
                                lx, ly = int(landmark[0]), int(landmark[1])
                                cv2.circle(annotated_image, (lx, ly), 3, (255, 255, 0), -1)
                    
                    # Draw regular landmarks if available
                    if person.landmarks:
                        for landmark in person.landmarks:
                            if len(landmark) >= 2:
                                lx, ly = int(landmark[0]), int(landmark[1])
                                cv2.circle(annotated_image, (lx, ly), 2, (0, 255, 255), -1)
            
            # 2. Face Recognition
            logger.info("   👤 Reconhecimento facial...")
            face_encodings = self.face_recognition_system.extract_face_encoding(image_path)
            
            if face_encodings:
                analysis_data['detections']['faces'] = []
                
                for i, face_encoding in enumerate(face_encodings):
                    face_data = {
                        'face_id': i,
                        'bbox': face_encoding.face_bbox,
                        'confidence': face_encoding.confidence,
                        'landmarks_count': len(face_encoding.landmarks) if face_encoding.landmarks else 0,
                        'has_encoding': face_encoding.face_encoding is not None
                    }
                    analysis_data['detections']['faces'].append(face_data)
                    
                    # Draw face bounding box
                    x, y, w, h = face_encoding.face_bbox
                    cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    # Add label
                    label = f"Face {i} ({face_encoding.confidence:.2f})"
                    cv2.putText(annotated_image, label, (x, y - 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # Draw face landmarks if available
                    if face_encoding.landmarks:
                        for landmark in face_encoding.landmarks:
                            if len(landmark) >= 2:
                                lx, ly = int(landmark[0]), int(landmark[1])
                                cv2.circle(annotated_image, (lx, ly), 1, (255, 0, 255), -1)
            
            # 3. Complete Feature Extraction
            logger.info("   📊 Extração completa de features...")
            features = self.feature_extractor.extract_features(image_path)
            
            if features:
                # Store key features
                key_features = {
                    'blur_score': features.get('sharpness_laplacian', 0),
                    'brightness': features.get('brightness_mean', 0),
                    'person_count': features.get('total_persons', 0),
                    'face_count': features.get('faces', 0),
                    'quality_score': features.get('unified_quality_score', 0),
                    'recommendation': features.get('unified_recommendation', 'unknown')
                }
                analysis_data['features'] = key_features
                
                # Add text overlay with key metrics
                y_offset = 30
                text_color = (255, 255, 255)
                bg_color = (0, 0, 0)
                
                metrics_text = [
                    f"Blur: {key_features['blur_score']:.1f}",
                    f"Brightness: {key_features['brightness']:.1f}",
                    f"Persons: {key_features['person_count']}",
                    f"Faces: {key_features['face_count']}",
                    f"Quality: {key_features['quality_score']:.2f}",
                    f"Rec: {key_features['recommendation']}"
                ]
                
                for i, text in enumerate(metrics_text):
                    y_pos = y_offset + (i * 25)
                    
                    # Background rectangle
                    (text_width, text_height), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_image, (10, y_pos - text_height - 5), 
                                 (15 + text_width, y_pos + 5), bg_color, -1)
                    
                    # Text
                    cv2.putText(annotated_image, text, (15, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            # 4. Save annotated image
            output_filename = f"{Path(filename).stem}_analysis.jpg"
            output_path = self.output_dir / output_filename
            
            cv2.imwrite(str(output_path), annotated_image)
            analysis_data['visualizations_created'].append(str(output_path))
            
            logger.info(f"   ✅ Visualização salva: {output_path}")
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"   ❌ Erro na análise de {filename}: {e}")
            return analysis_data
    
    def generate_analysis_report(self, all_analysis_data: list):
        """
        Generate comprehensive analysis report
        """
        if not all_analysis_data:
            return
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(all_analysis_data),
            'summary': {
                'total_persons_detected': 0,
                'total_faces_detected': 0,
                'images_with_persons': 0,
                'images_with_faces': 0,
                'average_blur_score': 0,
                'average_brightness': 0,
                'average_quality_score': 0
            },
            'detailed_results': all_analysis_data
        }
        
        # Calculate summary statistics
        blur_scores = []
        brightness_scores = []
        quality_scores = []
        
        for data in all_analysis_data:
            # Person stats
            persons = data.get('detections', {}).get('persons', [])
            faces = data.get('detections', {}).get('faces', [])
            
            report['summary']['total_persons_detected'] += len(persons)
            report['summary']['total_faces_detected'] += len(faces)
            
            if len(persons) > 0:
                report['summary']['images_with_persons'] += 1
            if len(faces) > 0:
                report['summary']['images_with_faces'] += 1
            
            # Feature stats
            features = data.get('features', {})
            if features.get('blur_score'):
                blur_scores.append(features['blur_score'])
            if features.get('brightness'):
                brightness_scores.append(features['brightness'])
            if features.get('quality_score'):
                quality_scores.append(features['quality_score'])
        
        # Calculate averages
        if blur_scores:
            report['summary']['average_blur_score'] = sum(blur_scores) / len(blur_scores)
        if brightness_scores:
            report['summary']['average_brightness'] = sum(brightness_scores) / len(brightness_scores)
        if quality_scores:
            report['summary']['average_quality_score'] = sum(quality_scores) / len(quality_scores)
        
        # Save report
        report_path = self.output_dir / "visual_analysis_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Relatório salvo: {report_path}")
        
        # Print summary
        logger.info("🎯 RESUMO DA ANÁLISE VISUAL:")
        logger.info(f"   📸 Imagens analisadas: {report['summary']['total_images']}")
        logger.info(f"   🎯 Pessoas detectadas: {report['summary']['total_persons_detected']}")
        logger.info(f"   👤 Rostos detectados: {report['summary']['total_faces_detected']}")
        logger.info(f"   📊 Blur médio: {report['summary']['average_blur_score']:.1f}")
        logger.info(f"   💡 Brilho médio: {report['summary']['average_brightness']:.1f}")
        logger.info(f"   ⭐ Qualidade média: {report['summary']['average_quality_score']:.2f}")
        
        return report
    
    def run_visual_analysis(self, image_count: int = 10):
        """
        Run complete visual analysis on test images
        """
        logger.info(f"🎯 INICIANDO ANÁLISE VISUAL - {image_count} IMAGENS")
        logger.info("=" * 60)
        
        # Initialize systems
        if not self.initialize_systems():
            return
        
        # Get test images
        input_dir = Path("data/input")
        if not input_dir.exists():
            logger.error(f"❌ Diretório não encontrado: {input_dir}")
            return
        
        # Find images
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        all_images = []
        
        for ext in image_extensions:
            all_images.extend(list(input_dir.glob(f"*{ext}")))
        
        if not all_images:
            logger.error("❌ Nenhuma imagem encontrada")
            return
        
        # Select images for analysis
        test_images = sorted(all_images)[:image_count]
        logger.info(f"📸 Selecionadas {len(test_images)} imagens para análise")
        
        # Analyze each image
        all_analysis_data = []
        
        for i, image_path in enumerate(test_images):
            logger.info(f"📋 [{i+1}/{len(test_images)}] Processando...")
            
            analysis_data = self.analyze_image_with_visualizations(str(image_path))
            if analysis_data:
                all_analysis_data.append(analysis_data)
        
        # Generate report
        logger.info("\n📊 Gerando relatório consolidado...")
        report = self.generate_analysis_report(all_analysis_data)
        
        logger.info("=" * 60)
        logger.info("🎉 ANÁLISE VISUAL COMPLETA!")
        logger.info(f"📁 Visualizações salvas em: {self.output_dir}")
        logger.info(f"📄 Relatório: {self.output_dir}/visual_analysis_report.json")
        
        return report


def main():
    """
    Execute visual analysis
    """
    generator = VisualAnalysisGenerator()
    generator.run_visual_analysis(image_count=15)


if __name__ == "__main__":
    main()
