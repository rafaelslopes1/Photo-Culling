#!/usr/bin/env python3
"""
Face Detection Debug Tool - Photo Culling System
Ferramenta de debug para detecção facial

This script specifically tests and debugs face detection issues
to identify why face detection is returning 0% success rate.
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
import mediapipe as mp

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


class FaceDetectionDebugger:
    """
    Debug tool for investigating face detection issues
    Ferramenta de debug para investigar problemas de detecção facial
    """
    
    def __init__(self):
        """Initialize debugger"""
        self.person_detector = PersonDetector()
        
        # Also initialize direct MediaPipe components for comparison
        self.mp_face_detection = None
        self.direct_face_detector = None
        
        try:
            # Try to import face detection
            import mediapipe.python.solutions.face_detection as face_detection
            self.mp_face_detection = face_detection
            self.direct_face_detector = face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.1  # Very low threshold for testing
            )
            print("✅ MediaPipe face detection carregado")
        except (ImportError, AttributeError) as e:
            print(f"⚠️ MediaPipe face_detection não disponível: {e}")
            # Try alternative approach
            try:
                import mediapipe as mp_alt
                if hasattr(mp_alt.solutions, 'face_detection'):
                    self.mp_face_detection = mp_alt.solutions.face_detection
                    self.direct_face_detector = self.mp_face_detection.FaceDetection(
                        model_selection=1,
                        min_detection_confidence=0.1
                    )
                    print("✅ MediaPipe face detection carregado (método alternativo)")
            except Exception as e2:
                print(f"❌ Não foi possível carregar face detection: {e2}")
        
        # Output directory
        self.output_dir = Path("data/quality/face_debug")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("🔍 Face Detection Debugger inicializado")
    
    def test_single_image_detailed(self, image_path: str) -> Dict[str, Any]:
        """Test face detection on single image with detailed analysis"""
        try:
            print(f"\n📸 Testando: {Path(image_path).name}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Erro: Não foi possível carregar {image_path}")
                return {'error': 'Failed to load image'}
            
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width = image.shape[:2]
            
            print(f"   📏 Resolução: {image_width}x{image_height}")
            
            # Test 1: Direct MediaPipe face detection with low threshold
            print(f"   🔍 Teste 1: MediaPipe direto (threshold 0.1)")
            direct_results = self.direct_face_detector.process(rgb_image)
            
            direct_faces = []
            if direct_results.detections:
                print(f"   ✅ {len(direct_results.detections)} face(s) detectada(s) diretamente")
                for i, detection in enumerate(direct_results.detections):
                    confidence = detection.score[0]
                    bbox = detection.location_data.relative_bounding_box
                    
                    x = int(bbox.xmin * image_width)
                    y = int(bbox.ymin * image_height)
                    w = int(bbox.width * image_width)
                    h = int(bbox.height * image_height)
                    
                    direct_faces.append({
                        'id': i,
                        'confidence': confidence,
                        'bbox': (x, y, w, h),
                        'relative_bbox': (bbox.xmin, bbox.ymin, bbox.width, bbox.height)
                    })
                    print(f"      Face {i+1}: conf={confidence:.3f}, bbox=({x},{y},{w},{h})")
            else:
                print(f"   ❌ Nenhuma face detectada diretamente")
            
            # Test 2: PersonDetector face detection
            print(f"   🔍 Teste 2: PersonDetector")
            person_results = self.person_detector.detect_persons_and_faces(image)
            
            print(f"   📊 Resultado PersonDetector:")
            print(f"      Pessoas: {person_results.get('total_persons', 0)}")
            print(f"      Faces: {person_results.get('total_faces', 0)}")
            
            faces_from_person_detector = person_results.get('faces', [])
            if faces_from_person_detector:
                print(f"   ✅ {len(faces_from_person_detector)} face(s) via PersonDetector")
                for face in faces_from_person_detector:
                    print(f"      Face ID {face.get('id')}: conf={face.get('confidence', 0):.3f}")
            else:
                print(f"   ❌ Nenhuma face via PersonDetector")
            
            # Test 3: Different MediaPipe configurations
            print(f"   🔍 Teste 3: Configurações alternativas")
            
            # Test with model_selection=0 (short-range model)
            short_range_detector = self.mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.1
            )
            
            short_results = short_range_detector.process(rgb_image)
            short_faces = len(short_results.detections) if short_results.detections else 0
            print(f"      Modelo short-range: {short_faces} face(s)")
            
            # Test with higher confidence threshold
            high_conf_detector = self.mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.7
            )
            
            high_results = high_conf_detector.process(rgb_image)
            high_faces = len(high_results.detections) if high_results.detections else 0
            print(f"      Threshold alto (0.7): {high_faces} face(s)")
            
            # Compile results
            result = {
                'filename': Path(image_path).name,
                'image_size': (image_width, image_height),
                'direct_mediapipe': {
                    'faces_count': len(direct_faces),
                    'faces': direct_faces
                },
                'person_detector': {
                    'total_persons': person_results.get('total_persons', 0),
                    'total_faces': person_results.get('total_faces', 0),
                    'faces': faces_from_person_detector
                },
                'alternative_configs': {
                    'short_range_faces': short_faces,
                    'high_confidence_faces': high_faces
                }
            }
            
            return result
            
        except Exception as e:
            print(f"   ❌ Erro no teste: {e}")
            return {'error': str(e), 'filename': Path(image_path).name}
    
    def create_debug_visualization(self, image_path: str, result: Dict[str, Any]):
        """Create visualization showing all detection attempts"""
        try:
            if 'error' in result:
                return
            
            # Load image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f"Debug Detecção Facial: {result['filename']}", fontsize=16, fontweight='bold')
            
            # 1. Original image
            ax1 = axes[0, 0]
            ax1.imshow(image_rgb)
            ax1.set_title("Imagem Original")
            ax1.axis('off')
            
            # 2. Direct MediaPipe detections
            ax2 = axes[0, 1]
            ax2.imshow(image_rgb)
            ax2.set_title(f"MediaPipe Direto ({result['direct_mediapipe']['faces_count']} faces)")
            ax2.axis('off')
            
            # Draw direct MediaPipe faces
            direct_faces = result['direct_mediapipe']['faces']
            for i, face in enumerate(direct_faces):
                x, y, w, h = face['bbox']
                rect = patches.Rectangle((x, y), w, h, linewidth=3,
                                       edgecolor='red', facecolor='none', alpha=0.8)
                ax2.add_patch(rect)
                ax2.text(x, y-10, f'Face {i+1}\nConf: {face["confidence"]:.2f}', 
                        color='red', fontweight='bold', fontsize=10)
            
            # 3. PersonDetector results
            ax3 = axes[1, 0]
            ax3.imshow(image_rgb)
            ax3.set_title(f"PersonDetector ({result['person_detector']['total_faces']} faces)")
            ax3.axis('off')
            
            # Draw PersonDetector faces
            person_faces = result['person_detector']['faces']
            for i, face in enumerate(person_faces):
                if 'bbox' in face:
                    x, y, w, h = face['bbox']
                    rect = patches.Rectangle((x, y), w, h, linewidth=3,
                                           edgecolor='blue', facecolor='none', alpha=0.8)
                    ax3.add_patch(rect)
                    ax3.text(x, y-10, f'Face {i+1}\nConf: {face.get("confidence", 0):.2f}', 
                            color='blue', fontweight='bold', fontsize=10)
            
            # 4. Summary statistics
            ax4 = axes[1, 1]
            
            # Create bar chart of detection results
            methods = ['MediaPipe\nDireto', 'Person\nDetector', 'Short\nRange', 'High\nConf']
            counts = [
                result['direct_mediapipe']['faces_count'],
                result['person_detector']['total_faces'],
                result['alternative_configs']['short_range_faces'],
                result['alternative_configs']['high_confidence_faces']
            ]
            
            colors = ['red', 'blue', 'green', 'orange']
            bars = ax4.bar(methods, counts, color=colors, alpha=0.7)
            ax4.set_title("Comparação de Métodos")
            ax4.set_ylabel("Faces Detectadas")
            ax4.set_ylim(0, max(max(counts), 1) + 1)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(count)}', ha='center', va='bottom', fontweight='bold')
            
            # Save visualization
            output_path = self.output_dir / f"debug_{Path(image_path).stem}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   💾 Debug salvo em: {output_path}")
            
        except Exception as e:
            print(f"   ❌ Erro na visualização: {e}")
    
    def run_debug_test(self, input_dir: str = "data/input", sample_size: int = 10):
        """Run debug test on sample images"""
        print(f"\n🔬 INICIANDO DEBUG DE DETECÇÃO FACIAL")
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
        import random
        sample_images = random.sample(all_images, min(sample_size, len(all_images)))
        
        print(f"🎯 Testando {len(sample_images)} imagens...")
        
        results = []
        
        for i, image_path in enumerate(sample_images, 1):
            print(f"\n[{i:2d}/{len(sample_images)}] Processando...")
            
            # Test image
            result = self.test_single_image_detailed(image_path)
            results.append(result)
            
            # Create visualization
            self.create_debug_visualization(image_path, result)
        
        # Generate summary report
        self.generate_debug_report(results)
        
        print(f"\n✅ DEBUG CONCLUÍDO!")
        print(f"📁 Resultados em: {self.output_dir}")
    
    def generate_debug_report(self, results: List[Dict[str, Any]]):
        """Generate comprehensive debug report"""
        try:
            # Filter valid results
            valid_results = [r for r in results if 'error' not in r]
            
            if not valid_results:
                print("❌ Nenhum resultado válido para relatório")
                return
            
            # Calculate statistics
            total_images = len(valid_results)
            
            # MediaPipe direct stats
            direct_face_counts = [r['direct_mediapipe']['faces_count'] for r in valid_results]
            direct_total = sum(direct_face_counts)
            direct_success_rate = sum(1 for c in direct_face_counts if c > 0) / total_images * 100
            
            # PersonDetector stats
            person_face_counts = [r['person_detector']['total_faces'] for r in valid_results]
            person_total = sum(person_face_counts)
            person_success_rate = sum(1 for c in person_face_counts if c > 0) / total_images * 100
            
            # Alternative config stats
            short_counts = [r['alternative_configs']['short_range_faces'] for r in valid_results]
            high_counts = [r['alternative_configs']['high_confidence_faces'] for r in valid_results]
            
            # Create report
            report = {
                'debug_summary': {
                    'total_images_tested': total_images,
                    'test_timestamp': str(np.datetime64('now'))
                },
                'detection_results': {
                    'direct_mediapipe': {
                        'total_faces_detected': direct_total,
                        'success_rate_percent': direct_success_rate,
                        'average_faces_per_image': direct_total / total_images,
                        'images_with_faces': sum(1 for c in direct_face_counts if c > 0)
                    },
                    'person_detector': {
                        'total_faces_detected': person_total,
                        'success_rate_percent': person_success_rate,
                        'average_faces_per_image': person_total / total_images,
                        'images_with_faces': sum(1 for c in person_face_counts if c > 0)
                    },
                    'alternative_configs': {
                        'short_range_total': sum(short_counts),
                        'high_confidence_total': sum(high_counts),
                        'short_range_success_rate': sum(1 for c in short_counts if c > 0) / total_images * 100,
                        'high_confidence_success_rate': sum(1 for c in high_counts if c > 0) / total_images * 100
                    }
                },
                'detailed_results': valid_results
            }
            
            # Save report
            report_path = self.output_dir / "face_detection_debug_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            # Print summary
            print(f"\n📊 RESUMO DO DEBUG:")
            print(f"   🎯 Imagens testadas: {total_images}")
            print(f"   🔴 MediaPipe direto: {direct_success_rate:.1f}% sucesso ({direct_total} faces)")
            print(f"   🔵 PersonDetector: {person_success_rate:.1f}% sucesso ({person_total} faces)")
            print(f"   🟢 Short-range: {sum(1 for c in short_counts if c > 0)/total_images*100:.1f}% sucesso")
            print(f"   🟠 High-confidence: {sum(1 for c in high_counts if c > 0)/total_images*100:.1f}% sucesso")
            
            print(f"\n💾 Relatório salvo em: {report_path}")
            
        except Exception as e:
            print(f"❌ Erro ao gerar relatório: {e}")


def main():
    """Main execution function"""
    print("🔍 FACE DETECTION DEBUG TOOL - PHOTO CULLING SYSTEM")
    print("Ferramenta de debug para detecção facial")
    print("=" * 80)
    
    # Initialize debugger
    debugger = FaceDetectionDebugger()
    
    # Run debug test
    debugger.run_debug_test(sample_size=10)


if __name__ == "__main__":
    main()
