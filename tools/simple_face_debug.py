#!/usr/bin/env python3
"""
Simple Face Detection Debug Tool - Photo Culling System
Ferramenta de debug simples para detecção facial

This script tests the existing PersonDetector to identify face detection issues.
"""

import os
import sys
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Any
import random

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Setup optimization and logging  
try:
    from src.utils.gpu_optimizer import MacM3Optimizer
    from src.utils.logging_config import enable_quiet_mode
    
    enable_quiet_mode()
    gpu_config, system_info = MacM3Optimizer.setup_quiet_and_optimized()
    print("🚀 Sistema otimizado para debug de detecção facial")
    
except Exception as e:
    print(f"⚠️ Otimização não disponível: {e}")

from src.core.person_detector import PersonDetector


class SimpleFaceDebugger:
    """
    Simple debug tool for face detection issues
    Ferramenta de debug simples para problemas de detecção facial
    """
    
    def __init__(self):
        """Initialize debugger"""
        self.person_detector = PersonDetector()
        
        # Output directory
        self.output_dir = Path("data/quality/face_debug_simple")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("🔍 Simple Face Detection Debugger inicializado")
    
    def test_face_detection_deep(self, image_path: str) -> Dict[str, Any]:
        """Deep test of face detection on single image"""
        try:
            print(f"\n📸 Análise profunda: {Path(image_path).name}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Erro: Não foi possível carregar {image_path}")
                return {'error': 'Failed to load image'}
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width = image.shape[:2]
            
            print(f"   📏 Resolução: {image_width}x{image_height}")
            print(f"   📊 Channels: {image.shape[2] if len(image.shape) > 2 else 'Grayscale'}")
            
            # Test PersonDetector with detailed analysis
            print(f"   🔍 Testando PersonDetector...")
            person_results = self.person_detector.detect_persons_and_faces(image)
            
            print(f"   📊 Resultado PersonDetector:")
            print(f"      Pessoas: {person_results.get('total_persons', 0)}")
            print(f"      Faces: {person_results.get('total_faces', 0)}")
            print(f"      Versão: {person_results.get('analysis_version', 'N/A')}")
            
            # Analyze each detected person
            persons = person_results.get('persons', [])
            faces = person_results.get('faces', [])
            
            print(f"   👥 Análise de pessoas detectadas:")
            if persons:
                for i, person in enumerate(persons):
                    if hasattr(person, '__dict__'):
                        print(f"      Pessoa {i+1}:")
                        print(f"        Dominância: {getattr(person, 'dominance_score', 'N/A')}")
                        print(f"        Confiança: {getattr(person, 'confidence', 'N/A')}")
                        print(f"        BBox: {getattr(person, 'bounding_box', 'N/A')}")
                        print(f"        Landmarks: {len(getattr(person, 'landmarks', []))} pontos")
                    elif isinstance(person, dict):
                        print(f"      Pessoa {i+1}: {person}")
            else:
                print(f"      ❌ Nenhuma pessoa detectada")
            
            print(f"   👤 Análise de faces detectadas:")
            if faces:
                for i, face in enumerate(faces):
                    print(f"      Face {i+1}: {face}")
            else:
                print(f"      ❌ Nenhuma face detectada")
            
            # Check if PersonDetector is properly initialized
            detector_status = {
                'initialized': getattr(self.person_detector, 'initialized', False),
                'has_pose': hasattr(self.person_detector, 'pose'),
                'has_face_detection': hasattr(self.person_detector, 'face_detection'),
            }
            
            print(f"   🔧 Status do detector:")
            for key, value in detector_status.items():
                print(f"      {key}: {value}")
            
            # Try to access MediaPipe components directly
            try:
                if hasattr(self.person_detector, 'face_detection'):
                    face_detector = self.person_detector.face_detection
                    print(f"   🎯 Testando face_detection diretamente...")
                    
                    # Process with face detector
                    face_results = face_detector.process(rgb_image)
                    
                    if face_results and hasattr(face_results, 'detections') and face_results.detections:
                        print(f"      ✅ {len(face_results.detections)} face(s) detectada(s) pelo MediaPipe")
                        
                        # Extract details
                        direct_faces = []
                        for detection in face_results.detections:
                            try:
                                confidence = detection.score[0] if hasattr(detection, 'score') else 0.0
                                
                                # Try to extract bbox
                                if hasattr(detection, 'location_data') and hasattr(detection.location_data, 'relative_bounding_box'):
                                    bbox = detection.location_data.relative_bounding_box
                                    x = int(bbox.xmin * image_width)
                                    y = int(bbox.ymin * image_height)
                                    w = int(bbox.width * image_width)
                                    h = int(bbox.height * image_height)
                                    
                                    direct_faces.append({
                                        'confidence': confidence,
                                        'bbox': (x, y, w, h)
                                    })
                                    print(f"        Face: conf={confidence:.3f}, bbox=({x},{y},{w},{h})")
                                
                            except Exception as e:
                                print(f"        ❌ Erro ao processar face: {e}")
                    else:
                        print(f"      ❌ Nenhuma face detectada pelo MediaPipe diretamente")
                        if face_results:
                            print(f"         face_results type: {type(face_results)}")
                            print(f"         face_results attrs: {dir(face_results)}")
                else:
                    print(f"   ❌ face_detection não disponível no PersonDetector")
                    
            except Exception as e:
                print(f"   ❌ Erro ao testar MediaPipe diretamente: {e}")
            
            # Compile detailed result
            result = {
                'filename': Path(image_path).name,
                'image_size': (image_width, image_height),
                'detector_status': detector_status,
                'person_detector_results': person_results,
                'analysis_details': {
                    'persons_found': len(persons),
                    'faces_found': len(faces),
                    'persons_data': [str(p) for p in persons],
                    'faces_data': faces
                }
            }
            
            return result
            
        except Exception as e:
            print(f"   ❌ Erro no teste: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'filename': Path(image_path).name}
    
    def create_simple_visualization(self, image_path: str, result: Dict[str, Any]):
        """Create simple visualization of detection results"""
        try:
            if 'error' in result:
                return
            
            # Load image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle(f"Debug Detecção: {result['filename']}", fontsize=16, fontweight='bold')
            
            # 1. Original image
            ax1 = axes[0]
            ax1.imshow(image_rgb)
            ax1.set_title("Imagem Original")
            ax1.axis('off')
            
            # 2. Detection results
            ax2 = axes[1]
            ax2.imshow(image_rgb)
            
            persons_found = result['analysis_details']['persons_found']
            faces_found = result['analysis_details']['faces_found']
            ax2.set_title(f"Detecções: {persons_found} pessoas, {faces_found} faces")
            ax2.axis('off')
            
            # Try to draw detected persons/faces
            person_results = result['person_detector_results']
            
            # Draw persons
            persons = person_results.get('persons', [])
            for i, person in enumerate(persons):
                try:
                    if hasattr(person, 'bounding_box'):
                        bbox = person.bounding_box
                    elif isinstance(person, dict) and 'bbox' in person:
                        bbox = person['bbox']
                    else:
                        continue
                    
                    x, y, w, h = bbox
                    rect = patches.Rectangle((x, y), w, h, linewidth=3,
                                           edgecolor='red', facecolor='none', alpha=0.8)
                    ax2.add_patch(rect)
                    ax2.text(x, y-10, f'Pessoa {i+1}', color='red', fontweight='bold')
                except Exception as e:
                    print(f"   ⚠️ Erro ao desenhar pessoa {i+1}: {e}")
            
            # Draw faces
            faces = person_results.get('faces', [])
            for i, face in enumerate(faces):
                try:
                    if 'bbox' in face:
                        x, y, w, h = face['bbox']
                        rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                               edgecolor='blue', facecolor='none', alpha=0.8)
                        ax2.add_patch(rect)
                        ax2.text(x, y-10, f'Face {i+1}', color='blue', fontweight='bold')
                except Exception as e:
                    print(f"   ⚠️ Erro ao desenhar face {i+1}: {e}")
            
            # Save visualization
            output_path = self.output_dir / f"debug_{Path(image_path).stem}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   💾 Debug salvo em: {output_path}")
            
        except Exception as e:
            print(f"   ❌ Erro na visualização: {e}")
    
    def run_simple_debug(self, input_dir: str = "data/input", sample_size: int = 5):
        """Run simple debug test"""
        print(f"\n🔬 DEBUG SIMPLES DE DETECÇÃO FACIAL")
        print("=" * 80)
        
        # Get sample images
        input_path = Path(input_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        all_images = [
            str(img) for img in input_path.rglob("*") 
            if img.suffix.lower() in image_extensions and img.stat().st_size > 10000
        ]
        
        if not all_images:
            print("❌ Nenhuma imagem encontrada")
            return
        
        # Take sample
        sample_images = random.sample(all_images, min(sample_size, len(all_images)))
        
        print(f"🎯 Testando {len(sample_images)} imagens...")
        
        results = []
        
        for i, image_path in enumerate(sample_images, 1):
            print(f"\n[{i:2d}/{len(sample_images)}] Processando...")
            
            # Test image
            result = self.test_face_detection_deep(image_path)
            results.append(result)
            
            # Create visualization
            self.create_simple_visualization(image_path, result)
        
        # Generate summary
        self.generate_simple_report(results)
        
        print(f"\n✅ DEBUG SIMPLES CONCLUÍDO!")
        print(f"📁 Resultados em: {self.output_dir}")
    
    def generate_simple_report(self, results: List[Dict[str, Any]]):
        """Generate simple debug report"""
        try:
            valid_results = [r for r in results if 'error' not in r]
            
            if not valid_results:
                print("❌ Nenhum resultado válido")
                return
            
            # Calculate basic statistics
            total_images = len(valid_results)
            
            person_counts = [r['analysis_details']['persons_found'] for r in valid_results]
            face_counts = [r['analysis_details']['faces_found'] for r in valid_results]
            
            total_persons = sum(person_counts)
            total_faces = sum(face_counts)
            
            images_with_persons = sum(1 for c in person_counts if c > 0)
            images_with_faces = sum(1 for c in face_counts if c > 0)
            
            # Create report
            report = {
                'debug_summary': {
                    'total_images': total_images,
                    'images_with_persons': images_with_persons,
                    'images_with_faces': images_with_faces,
                    'person_success_rate': images_with_persons / total_images * 100,
                    'face_success_rate': images_with_faces / total_images * 100,
                    'total_persons_detected': total_persons,
                    'total_faces_detected': total_faces,
                    'avg_persons_per_image': total_persons / total_images,
                    'avg_faces_per_image': total_faces / total_images
                },
                'detailed_results': valid_results
            }
            
            # Save report
            report_path = self.output_dir / "simple_debug_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            # Print summary
            print(f"\n📊 RESUMO DO DEBUG SIMPLES:")
            print(f"   🎯 Imagens testadas: {total_images}")
            print(f"   👥 Taxa detecção pessoas: {images_with_persons/total_images*100:.1f}% ({images_with_persons}/{total_images})")
            print(f"   👤 Taxa detecção faces: {images_with_faces/total_images*100:.1f}% ({images_with_faces}/{total_images})")
            print(f"   📊 Total pessoas: {total_persons}")
            print(f"   📊 Total faces: {total_faces}")
            print(f"   📈 Média pessoas/imagem: {total_persons/total_images:.1f}")
            print(f"   📈 Média faces/imagem: {total_faces/total_images:.1f}")
            
            print(f"\n💾 Relatório salvo em: {report_path}")
            
        except Exception as e:
            print(f"❌ Erro ao gerar relatório: {e}")


def main():
    """Main execution function"""
    print("🔍 SIMPLE FACE DETECTION DEBUG - PHOTO CULLING SYSTEM")
    print("Debug simples para detecção facial")
    print("=" * 80)
    
    # Initialize debugger
    debugger = SimpleFaceDebugger()
    
    # Run debug test
    debugger.run_simple_debug(sample_size=5)


if __name__ == "__main__":
    main()
