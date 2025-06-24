#!/usr/bin/env python3
"""
Advanced Multi-Person Detection Test
Busca por imagens com múltiplas pessoas e cria visualização especial
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

def find_multi_person_images(max_scan=100):
    """Scan images to find ones with multiple people"""
    print("🔍 Buscando imagens com múltiplas pessoas...")
    
    person_detector = PersonDetector()
    input_dir = Path("data/input")
    image_files = list(input_dir.glob("*.JPG"))[:max_scan]
    
    multi_person_images = []
    
    for i, image_path in enumerate(image_files):
        if i % 10 == 0:
            print(f"   Verificando {i+1}/{len(image_files)}...")
        
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
                
            result = person_detector.detect_persons_and_faces(image)
            person_count = result.get('total_persons', 0)
            face_count = result.get('total_faces', 0)
            
            if person_count > 1 or face_count > 1:
                multi_person_images.append({
                    'path': str(image_path),
                    'persons': person_count,
                    'faces': face_count,
                    'result': result
                })
                print(f"   ✅ Encontrada: {image_path.name} - {person_count} pessoas, {face_count} faces")
                
        except Exception as e:
            continue
    
    return multi_person_images

def create_advanced_visualization(multi_person_images):
    """Create detailed visualization for multi-person images"""
    print(f"\n🎨 Criando visualização avançada para {len(multi_person_images)} imagens...")
    
    if not multi_person_images:
        print("❌ Nenhuma imagem com múltiplas pessoas encontrada.")
        return
    
    # Create output directory
    output_dir = Path("data/quality/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_data in multi_person_images[:6]:  # Limit to 6 images for visualization
        image_path = img_data['path']
        detection_result = img_data['result']
        
        print(f"📸 Processando: {Path(image_path).name}")
        
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create detailed visualization
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Original image
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Imagem Original', fontsize=16, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Image with all detections
        vis_image = draw_detailed_detections(image, detection_result)
        vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        axes[0, 1].imshow(vis_image_rgb)
        axes[0, 1].set_title(f'Todas as Detecções ({img_data["persons"]} pessoas, {img_data["faces"]} faces)', 
                           fontsize=16, fontweight='bold')
        axes[0, 1].axis('off')
        
        # Person analysis chart
        create_person_analysis_chart(axes[1, 0], detection_result)
        
        # Detection details
        create_detection_details(axes[1, 1], detection_result, Path(image_path).name)
        
        plt.tight_layout()
        
        # Save detailed visualization
        output_filename = f"{Path(image_path).stem}_multi_person_analysis.png"
        output_path = output_dir / output_filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Salvo: {output_path}")

def draw_detailed_detections(image, detection_result):
    """Draw detailed detection visualization"""
    vis_image = image.copy()
    
    # Draw persons with different colors
    colors = [(0, 255, 0), (255, 165, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0)]
    
    if 'persons' in detection_result and detection_result['persons']:
        for i, person in enumerate(detection_result['persons']):
            color = colors[i % len(colors)]
            bbox = person.bounding_box
            x, y, w, h = bbox
            
            # Draw person box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 4)
            
            # Add detailed label
            label = f"P{i+1}: Dom={person.dominance_score:.3f}"
            label += f" Area={person.area_ratio:.3f}"
            
            # Background for text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(vis_image, (x, y-30), (x + text_size[0] + 10, y), color, -1)
            cv2.putText(vis_image, label, (x + 5, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Draw faces with blue boxes
    if 'faces' in detection_result and detection_result['faces']:
        for i, face in enumerate(detection_result['faces']):
            if hasattr(face, 'bounding_box'):
                x, y, w, h = face.bounding_box
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(vis_image, f"F{i+1}", (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return vis_image

def create_person_analysis_chart(ax, detection_result):
    """Create person analysis chart"""
    ax.set_title('Análise de Pessoas Detectadas', fontsize=14, fontweight='bold')
    
    if 'persons' not in detection_result or not detection_result['persons']:
        ax.text(0.5, 0.5, 'Nenhuma pessoa detectada', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axis('off')
        return
    
    persons = detection_result['persons']
    
    # Create metrics
    person_ids = [f"Pessoa {i+1}" for i in range(len(persons))]
    dominance_scores = [p.dominance_score for p in persons]
    area_ratios = [p.area_ratio for p in persons]
    centrality_scores = [p.centrality for p in persons]
    
    # Bar chart
    x = np.arange(len(person_ids))
    width = 0.25
    
    ax.bar(x - width, dominance_scores, width, label='Dominância', alpha=0.8, color='lightgreen')
    ax.bar(x, area_ratios, width, label='Área Relativa', alpha=0.8, color='lightblue')
    ax.bar(x + width, centrality_scores, width, label='Centralidade', alpha=0.8, color='lightcoral')
    
    ax.set_ylabel('Score')
    ax.set_xlabel('Pessoas')
    ax.set_xticks(x)
    ax.set_xticklabels(person_ids)
    ax.legend()
    ax.grid(True, alpha=0.3)

def create_detection_details(ax, detection_result, filename):
    """Create detection details text"""
    ax.set_title('Detalhes da Detecção', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Compile detailed information
    details = f"📁 Arquivo: {filename}\n\n"
    details += f"📊 ESTATÍSTICAS GERAIS:\n"
    details += f"• Pessoas detectadas: {detection_result.get('total_persons', 0)}\n"
    details += f"• Faces detectadas: {detection_result.get('total_faces', 0)}\n"
    details += f"• Versão de análise: {detection_result.get('analysis_version', 'N/A')}\n\n"
    
    if 'persons' in detection_result and detection_result['persons']:
        details += f"👥 DETALHES DAS PESSOAS:\n"
        for i, person in enumerate(detection_result['persons']):
            details += f"Pessoa {i+1}:\n"
            details += f"  • Dominância: {person.dominance_score:.3f}\n"
            details += f"  • Área relativa: {person.area_ratio:.3f}\n"
            details += f"  • Centralidade: {person.centrality:.3f}\n"
            details += f"  • Posição: {person.bounding_box}\n\n"
    
    if 'dominant_person' in detection_result and detection_result['dominant_person']:
        dominant = detection_result['dominant_person']
        details += f"⭐ PESSOA DOMINANTE:\n"
        details += f"• Score: {dominant.dominance_score:.3f}\n"
        details += f"• ID: {dominant.person_id}\n"
    
    ax.text(0.05, 0.95, details, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", alpha=0.8))

def main():
    print("🎬 TESTE AVANÇADO DE DETECÇÃO DE MÚLTIPLAS PESSOAS")
    print("=" * 55)
    
    # Find images with multiple people
    multi_person_images = find_multi_person_images()
    
    if not multi_person_images:
        print("\n🤔 Não foram encontradas imagens com múltiplas pessoas.")
        print("   Executando teste com imagens de uma pessoa...")
        
        # Run regular test if no multi-person images found
        from test_visual_detection import test_visual_detection
        test_visual_detection(max_images=6)
        return
    
    print(f"\n🎉 Encontradas {len(multi_person_images)} imagens com múltiplas pessoas!")
    
    # Create advanced visualizations
    create_advanced_visualization(multi_person_images)
    
    # Print summary
    print("\n" + "="*55)
    print("📋 RESUMO - IMAGENS COM MÚLTIPLAS PESSOAS")
    print("="*55)
    
    for img_data in multi_person_images:
        filename = Path(img_data['path']).name
        print(f"📸 {filename:<30} | 👥 {img_data['persons']:2d} pessoas | 😊 {img_data['faces']:2d} faces")
    
    print(f"\n🎯 Total: {len(multi_person_images)} imagens com múltiplas pessoas")
    print("📁 Visualizações salvas em: data/quality/visualizations/")
    print("="*55)

if __name__ == "__main__":
    main()
