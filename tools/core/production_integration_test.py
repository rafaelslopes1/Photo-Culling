#!/usr/bin/env python3
"""
Production Integration Test - Teste de Integração de Produção
Testa o pipeline completo com as correções integradas
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import logging
import json
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.person_detector import PersonDetector
from src.core.feature_extractor import FeatureExtractor
from src.core.face_recognition_system import FaceRecognitionSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionIntegrationTester:
    """
    Testa a integração completa do pipeline de produção
    """
    
    def __init__(self):
        self.person_detector = None
        self.feature_extractor = None
        self.face_recognition_system = None
        
    def initialize(self):
        """Initialize all systems"""
        try:
            logger.info("🔧 Inicializando sistemas de produção...")
            
            self.person_detector = PersonDetector()
            self.feature_extractor = FeatureExtractor()
            self.face_recognition_system = FaceRecognitionSystem()
            
            logger.info("✅ Todos os sistemas inicializados com sucesso")
            return True
        except Exception as e:
            logger.error(f"❌ Erro na inicialização: {e}")
            return False
    
    def test_integrated_pipeline(self, image_path: str) -> Dict:
        """
        Test the complete integrated pipeline
        """
        logger.info(f"🔍 TESTANDO PIPELINE INTEGRADO: {os.path.basename(image_path)}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error("❌ Não foi possível carregar a imagem")
            return {}
        
        results = {
            'image_path': image_path,
            'image_name': os.path.basename(image_path)
        }
        
        # 1. Person Detection (with improvements)
        logger.info("   👥 Detecção de pessoas e faces...")
        try:
            person_result = self.person_detector.detect_persons_and_faces(image)
            
            results['person_detection'] = {
                'total_persons': person_result.get('total_persons', 0),
                'total_faces': person_result.get('total_faces', 0),
                'persons_found': len(person_result.get('persons', [])),
                'faces_found': len(person_result.get('faces', [])),
                'has_dominant_person': person_result.get('dominant_person') is not None
            }
            
            if person_result.get('dominant_person'):
                dp = person_result['dominant_person']
                results['dominant_person'] = {
                    'dominance_score': float(dp.dominance_score),
                    'confidence': float(dp.confidence),
                    'area_ratio': float(dp.area_ratio),
                    'centrality': float(dp.centrality),
                    'local_sharpness': float(dp.local_sharpness),
                    'bbox': dp.bounding_box,
                    'landmarks_count': len(dp.landmarks or [])
                }
            
            logger.info(f"      ✅ {results['person_detection']['total_persons']} pessoas, {results['person_detection']['total_faces']} faces")
            
        except Exception as e:
            logger.error(f"      ❌ Erro na detecção de pessoas: {e}")
            results['person_detection'] = {'error': str(e)}
        
        # 2. Feature Extraction (with correct metrics)
        logger.info("   📊 Extração de características...")
        try:
            # Use the corrected quality metrics method
            quality_metrics = self.feature_extractor._extract_quality_metrics(image)
            
            results['quality_metrics'] = {
                'sharpness_laplacian': quality_metrics['sharpness_laplacian'],
                'sharpness_sobel': quality_metrics['sharpness_sobel'],
                'brightness_mean': quality_metrics['brightness_mean'],
                'brightness_std': quality_metrics['brightness_std'],
                'contrast_rms': quality_metrics['contrast_rms'],
                'saturation_mean': quality_metrics['saturation_mean'],
                'noise_level': quality_metrics['noise_level']
            }
            
            # Quality assessment
            blur_score = quality_metrics['sharpness_laplacian']
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
            
            results['quality_assessment'] = {
                'quality_level': quality_level,
                'recommendation': recommendation,
                'quality_score': min(1.0, blur_score / 200.0)
            }
            
            logger.info(f"      ✅ Qualidade: {quality_level} (blur: {blur_score:.1f})")
            
        except Exception as e:
            logger.error(f"      ❌ Erro na extração de características: {e}")
            results['quality_metrics'] = {'error': str(e)}
        
        # 3. Face Recognition (if faces found)
        if results.get('person_detection', {}).get('total_faces', 0) > 0:
            logger.info("   🧠 Reconhecimento facial...")
            try:
                # Extract face encodings
                face_encodings = []
                faces = person_result.get('faces', [])
                
                for i, face in enumerate(faces):
                    try:
                        bbox = face.get('bbox', [])
                        if len(bbox) == 4:
                            x, y, w, h = bbox
                            face_roi = image[y:y+h, x:x+w]
                            
                            if face_roi.size > 0:
                                encoding = self.face_recognition_system.extract_face_encoding(face_roi)
                                if encoding is not None:
                                    face_encodings.append({
                                        'face_id': i,
                                        'encoding_length': len(encoding),
                                        'has_encoding': True
                                    })
                    except Exception as e:
                        logger.debug(f"        Face {i} encoding falhou: {e}")
                
                results['face_recognition'] = {
                    'encodings_extracted': len(face_encodings),
                    'encoding_details': face_encodings
                }
                
                logger.info(f"      ✅ {len(face_encodings)} encodings faciais extraídos")
                
            except Exception as e:
                logger.error(f"      ❌ Erro no reconhecimento facial: {e}")
                results['face_recognition'] = {'error': str(e)}
        
        return results
    
    def test_batch_integration(self, image_paths: List[str]) -> Dict:
        """
        Test integration on a batch of images
        """
        logger.info(f"🚀 TESTE DE INTEGRAÇÃO EM LOTE: {len(image_paths)} imagens")
        logger.info("=" * 60)
        
        batch_results = {
            'total_images': len(image_paths),
            'processed_images': 0,
            'successful_processes': 0,
            'failed_processes': 0,
            'results': [],
            'summary_stats': {}
        }
        
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"\n[{i}/{len(image_paths)}] Processando: {os.path.basename(image_path)}")
            
            try:
                result = self.test_integrated_pipeline(image_path)
                
                if result:
                    batch_results['results'].append(result)
                    batch_results['successful_processes'] += 1
                    logger.info(f"      ✅ Processamento bem-sucedido")
                else:
                    batch_results['failed_processes'] += 1
                    logger.error(f"      ❌ Processamento falhou")
                
                batch_results['processed_images'] += 1
                
            except Exception as e:
                logger.error(f"      ❌ Erro no processamento: {e}")
                batch_results['failed_processes'] += 1
        
        # Calculate summary statistics
        if batch_results['results']:
            total_persons = sum(r.get('person_detection', {}).get('total_persons', 0) for r in batch_results['results'])
            total_faces = sum(r.get('person_detection', {}).get('total_faces', 0) for r in batch_results['results'])
            
            blur_scores = [r.get('quality_metrics', {}).get('sharpness_laplacian', 0) for r in batch_results['results']]
            blur_scores = [b for b in blur_scores if b > 0]
            
            quality_levels = [r.get('quality_assessment', {}).get('quality_level', 'unknown') for r in batch_results['results']]
            quality_counts = {}
            for level in quality_levels:
                quality_counts[level] = quality_counts.get(level, 0) + 1
            
            batch_results['summary_stats'] = {
                'avg_persons_per_image': total_persons / len(batch_results['results']),
                'avg_faces_per_image': total_faces / len(batch_results['results']),
                'avg_blur_score': sum(blur_scores) / len(blur_scores) if blur_scores else 0,
                'quality_distribution': quality_counts
            }
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 RESUMO DO TESTE EM LOTE:")
        logger.info(f"   Imagens processadas: {batch_results['processed_images']}/{batch_results['total_images']}")
        logger.info(f"   Sucessos: {batch_results['successful_processes']}")
        logger.info(f"   Falhas: {batch_results['failed_processes']}")
        
        if batch_results['summary_stats']:
            stats = batch_results['summary_stats']
            logger.info(f"   Pessoas por imagem (média): {stats['avg_persons_per_image']:.1f}")
            logger.info(f"   Faces por imagem (média): {stats['avg_faces_per_image']:.1f}")
            logger.info(f"   Blur score médio: {stats['avg_blur_score']:.1f}")
            logger.info(f"   Distribuição de qualidade: {stats['quality_distribution']}")
        
        return batch_results


def main():
    """Run production integration tests"""
    tester = ProductionIntegrationTester()
    
    if not tester.initialize():
        logger.error("❌ Falha na inicialização dos sistemas")
        return
    
    # Test images
    test_images = [
        "data/input/IMG_0252.JPG",  # Multi-face image
        "data/input/IMG_0001.JPG",  # Single person
        "data/input/IMG_0304.JPG",  # Multiple people
        "data/input/IMG_0339.JPG",
        "data/input/IMG_0400.JPG"
    ]
    
    # Filter existing images
    existing_images = [img for img in test_images if os.path.exists(img)]
    
    if not existing_images:
        logger.error("❌ Nenhuma imagem de teste encontrada")
        return
    
    # Run batch test
    batch_results = tester.test_batch_integration(existing_images)
    
    # Save results
    output_dir = Path("data/analysis_results/production_integration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"integration_test_{int(time.time())}.json"
    
    # Make results JSON serializable
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        return obj
    
    batch_results = make_serializable(batch_results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✅ Resultados salvos em: {output_file}")
    logger.info("🎉 TESTE DE INTEGRAÇÃO CONCLUÍDO!")


if __name__ == "__main__":
    import time
    main()
