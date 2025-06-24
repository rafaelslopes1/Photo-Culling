#!/usr/bin/env python3
"""
Visual Person Detection Test
Tests multi-person detection with image visualization
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from core.person_detector import PersonDetector
    from core.feature_extractor import FeatureExtractor
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    sys.exit(1)

def draw_person_detections(image, detection_result):
    """Draw person detection boxes and info on image"""
    vis_image = image.copy()
    
    # Draw person bounding boxes
    if 'persons' in detection_result and detection_result['persons']:
        for i, person in enumerate(detection_result['persons']):
            # Get bounding box from person detection
            bbox = person.bounding_box
            x, y, w, h = bbox
            x2, y2 = x + w, y + h
            
            # Draw bounding box in green
            cv2.rectangle(vis_image, (x, y), (x2, y2), (0, 255, 0), 3)
            
            # Add person label with dominance score
            label = f"Person {i+1} (Dom: {person.dominance_score:.2f})"
            cv2.putText(vis_image, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw face detections if available
    if 'faces' in detection_result and detection_result['faces']:
        for i, face in enumerate(detection_result['faces']):
            # Face detection should have bounding_box attribute
            if hasattr(face, 'bounding_box'):
                x, y, w, h = face.bounding_box
                x2, y2 = x + w, y + h
                
                # Draw face box in blue
                cv2.rectangle(vis_image, (x, y), (x2, y2), (255, 0, 0), 2)
                
                # Add face label
                cv2.putText(vis_image, f"Face {i+1}", (x, y2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return vis_image

def test_visual_detection(image_paths=None, max_images=6):
    """Test person detection with visual output"""
    print("🎯 Iniciando teste visual de detecção de pessoas...")
    
    # Initialize detectors
    print("📦 Inicializando detectores...")
    person_detector = PersonDetector()
    feature_extractor = FeatureExtractor()
    
    # Get test images
    if not image_paths:
        input_dir = Path("data/input")
        image_files = list(input_dir.glob("*.JPG"))[:max_images]
        image_paths = [str(f) for f in image_files]
    
    if not image_paths:
        print("❌ Nenhuma imagem encontrada!")
        return
    
    print(f"🖼️  Processando {len(image_paths)} imagens...")
    
    # Create output directory
    output_dir = Path("data/quality/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    results = []
    
    for i, image_path in enumerate(image_paths):
        print(f"\n📸 Processando imagem {i+1}/{len(image_paths)}: {Path(image_path).name}")
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Erro ao carregar imagem: {image_path}")
                continue
            
            # Detect persons
            detection_result = person_detector.detect_persons_and_faces(image)
            
            # Extract full features for additional context
            features = feature_extractor.extract_features(image_path)
            
            # Create visualization
            vis_image = draw_person_detections(image, detection_result)
            
            # Convert to RGB for matplotlib
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
            
            # Create side-by-side comparison
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Original image
            axes[0].imshow(image_rgb)
            axes[0].set_title('Imagem Original', fontsize=14)
            axes[0].axis('off')
            
            # Detection visualization
            axes[1].imshow(vis_image_rgb)
            axes[1].set_title(f'Detecções: {detection_result.get("total_persons", 0)} pessoas', fontsize=14)
            axes[1].axis('off')
            
            # Add detection information
            dominant_score = 0
            if detection_result.get('dominant_person'):
                dominant_score = detection_result['dominant_person'].dominance_score
            
            info_text = f"""
Arquivo: {Path(image_path).name}
Pessoas detectadas: {detection_result.get('total_persons', 0)}
Faces detectadas: {detection_result.get('total_faces', 0)}
Pessoa dominante (score): {dominant_score:.3f}
Versão de análise: {detection_result.get('analysis_version', 'unknown')}
            """
            
            plt.figtext(0.02, 0.15, info_text.strip(), fontsize=11,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))
            
            plt.tight_layout()
            
            # Save visualization
            output_filename = f"{Path(image_path).stem}_detection_test.png"
            output_path = output_dir / output_filename
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Store results
            dominant_score = 0
            if detection_result.get('dominant_person'):
                dominant_score = detection_result['dominant_person'].dominance_score
                
            result = {
                'image': Path(image_path).name,
                'person_count': detection_result.get('total_persons', 0),
                'face_count': detection_result.get('total_faces', 0),
                'dominant_score': dominant_score,
                'analysis_version': detection_result.get('analysis_version', 'unknown'),
                'output_file': str(output_path),
                'features': {
                    'blur_score': features.get('sharpness_laplacian', 0),
                    'brightness': features.get('brightness_mean', 0),
                    'exposure_level': features.get('exposure_level', 'unknown')
                }
            }
            results.append(result)
            
            print(f"✅ {detection_result.get('total_persons', 0)} pessoas detectadas")
            print(f"📁 Visualização salva: {output_path}")
            
        except Exception as e:
            print(f"❌ Erro ao processar {image_path}: {e}")
            continue
    
    # Create summary visualization
    if results:
        create_summary_visualization(results, output_dir)
    
    # Print summary
    print_test_summary(results)
    
    return results

def create_summary_visualization(results, output_dir):
    """Create summary chart of detection results"""
    print("\n📊 Criando visualização de resumo...")
    
    # Extract data for plotting
    images = [r['image'] for r in results]
    person_counts = [r['person_count'] for r in results]
    face_counts = [r['face_count'] for r in results]
    dominant_scores = [r['dominant_score'] for r in results]
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Person count distribution
    axes[0, 0].bar(range(len(images)), person_counts, color='lightblue', alpha=0.7)
    axes[0, 0].set_title('Pessoas Detectadas por Imagem')
    axes[0, 0].set_ylabel('Número de Pessoas')
    axes[0, 0].set_xticks(range(len(images)))
    axes[0, 0].set_xticklabels([img[:10] + '...' if len(img) > 10 else img for img in images], rotation=45)
    
    # Face vs Person comparison
    x_pos = np.arange(len(images))
    width = 0.35
    axes[0, 1].bar(x_pos - width/2, person_counts, width, label='Pessoas', color='lightgreen', alpha=0.7)
    axes[0, 1].bar(x_pos + width/2, face_counts, width, label='Faces', color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('Pessoas vs Faces Detectadas')
    axes[0, 1].set_ylabel('Contagem')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels([img[:8] + '...' if len(img) > 8 else img for img in images], rotation=45)
    axes[0, 1].legend()
    
    # Dominant person scores
    axes[1, 0].plot(range(len(images)), dominant_scores, 'o-', color='purple', alpha=0.7)
    axes[1, 0].set_title('Score da Pessoa Dominante')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xlabel('Imagem')
    axes[1, 0].set_xticks(range(len(images)))
    axes[1, 0].set_xticklabels([img[:8] + '...' if len(img) > 8 else img for img in images], rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Summary statistics
    total_people = sum(person_counts)
    avg_people = total_people / len(results) if results else 0
    avg_dominant_score = sum(dominant_scores) / len(results) if results else 0
    
    stats_text = f"""
ESTATÍSTICAS DO TESTE:
• Imagens processadas: {len(results)}
• Total de pessoas detectadas: {total_people}
• Média de pessoas por imagem: {avg_people:.2f}
• Score médio pessoa dominante: {avg_dominant_score:.3f}
• Taxa de sucesso: {len([r for r in results if r['person_count'] > 0]) / len(results) * 100:.1f}%
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text.strip(), fontsize=12, 
                    verticalalignment='center', transform=axes[1, 1].transAxes,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f4fd", alpha=0.8))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save summary
    summary_path = output_dir / 'detection_test_summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Resumo salvo: {summary_path}")

def print_test_summary(results):
    """Print detailed test summary"""
    print("\n" + "="*60)
    print("📋 RESUMO DO TESTE DE DETECÇÃO VISUAL")
    print("="*60)
    
    if not results:
        print("❌ Nenhum resultado para exibir.")
        return
    
    total_people = sum(r['person_count'] for r in results)
    total_faces = sum(r['face_count'] for r in results)
    avg_people = total_people / len(results)
    avg_faces = total_faces / len(results)
    success_rate = len([r for r in results if r['person_count'] > 0]) / len(results) * 100
    
    print(f"📊 Imagens processadas: {len(results)}")
    print(f"👥 Total de pessoas detectadas: {total_people}")
    print(f"😊 Total de faces detectadas: {total_faces}")
    print(f"📈 Média de pessoas por imagem: {avg_people:.2f}")
    print(f"📈 Média de faces por imagem: {avg_faces:.2f}")
    print(f"✅ Taxa de sucesso na detecção: {success_rate:.1f}%")
    
    print("\n📋 DETALHES POR IMAGEM:")
    print("-" * 60)
    for result in results:
        print(f"📸 {result['image']:<20} | "
              f"👥 {result['person_count']:2d} pessoas | "
              f"😊 {result['face_count']:2d} faces | "
              f"⭐ {result['dominant_score']:.3f} score")
    
    print("\n🎯 Status geral:", end=" ")
    if success_rate >= 80:
        print("🎉 EXCELENTE")
    elif success_rate >= 60:
        print("✅ BOM")
    elif success_rate >= 40:
        print("⚠️  RAZOÁVEL")
    else:
        print("❌ PRECISA MELHORAR")
    
    print("="*60)

if __name__ == "__main__":
    print("🎬 TESTE VISUAL DE DETECÇÃO DE MÚLTIPLAS PESSOAS")
    print("=" * 50)
    
    # Run the test
    results = test_visual_detection()
    
    if results:
        print(f"\n🎉 Teste concluído! {len(results)} imagens processadas.")
        print("📁 Visualizações salvas em: data/quality/visualizations/")
    else:
        print("\n❌ Nenhuma imagem foi processada com sucesso.")
