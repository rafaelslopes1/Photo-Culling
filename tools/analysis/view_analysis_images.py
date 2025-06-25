#!/usr/bin/env python3
"""
Image Viewer - Visualizador de Imagens com Análise
Abre as imagens analisadas para visualização das detecções
"""

import os
import sys
from pathlib import Path
import json

def show_analysis_summary():
    """
    Show analysis summary from the report
    """
    report_path = Path("data/analysis_results/visual_analysis/visual_analysis_report.json")
    
    if not report_path.exists():
        print("❌ Relatório não encontrado. Execute primeiro o visual_analysis_generator.py")
        return
    
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)
    
    summary = report.get('summary', {})
    
    print("🎯 RESUMO DA ANÁLISE VISUAL")
    print("=" * 50)
    print(f"📸 Imagens analisadas: {report.get('total_images', 0)}")
    print(f"🎯 Pessoas detectadas: {summary.get('total_persons_detected', 0)}")
    print(f"👤 Rostos detectados: {summary.get('total_faces_detected', 0)}")
    print(f"📊 Blur médio: {summary.get('average_blur_score', 0):.1f}")
    print(f"💡 Brilho médio: {summary.get('average_brightness', 0):.1f}")
    print(f"⭐ Qualidade média: {summary.get('average_quality_score', 0):.2f}")
    print()
    
    # Show individual results
    print("📋 DETALHES POR IMAGEM:")
    print("-" * 50)
    
    for result in report.get('detailed_results', []):
        filename = result.get('filename', 'unknown')
        detections = result.get('detections', {})
        features = result.get('features', {})
        
        persons = detections.get('persons', [])
        faces = detections.get('faces', [])
        
        print(f"📸 {filename}")
        print(f"   🎯 Pessoas: {len(persons)} | 👤 Rostos: {len(faces)}")
        print(f"   📊 Blur: {features.get('blur_score', 0):.1f} | 💡 Brilho: {features.get('brightness', 0):.1f}")
        print(f"   ⭐ Qualidade: {features.get('quality_score', 0):.2f} | 🏷️ Rec: {features.get('recommendation', 'N/A')}")
        
        # Show landmarks info
        if persons:
            for i, person in enumerate(persons):
                landmarks_count = person.get('landmarks_count', 0)
                pose_landmarks_count = person.get('pose_landmarks_count', 0)
                print(f"      👤 Pessoa {i}: {landmarks_count} landmarks gerais, {pose_landmarks_count} pose landmarks")
        
        if faces:
            for i, face in enumerate(faces):
                landmarks_count = face.get('landmarks_count', 0)
                has_encoding = face.get('has_encoding', False)
                confidence = face.get('confidence', 0)
                print(f"      😊 Rosto {i}: {landmarks_count} landmarks, encoding: {'✅' if has_encoding else '❌'}, conf: {confidence:.2f}")
        
        print()

def list_generated_images():
    """
    List all generated analysis images
    """
    analysis_dir = Path("data/analysis_results/visual_analysis")
    
    if not analysis_dir.exists():
        print("❌ Diretório de análise não encontrado")
        return []
    
    images = list(analysis_dir.glob("*_analysis.jpg"))
    
    if not images:
        print("❌ Nenhuma imagem de análise encontrada")
        return []
    
    print(f"🖼️ IMAGENS DE ANÁLISE GERADAS ({len(images)}):")
    print("=" * 60)
    
    for i, image in enumerate(sorted(images)):
        print(f"{i+1:2d}. {image.name}")
    
    print()
    return sorted(images)

def open_images_in_default_viewer():
    """
    Open images in default system viewer
    """
    images = list_generated_images()
    
    if not images:
        return
    
    print("🚀 Abrindo imagens no visualizador padrão do sistema...")
    
    for image in images:
        try:
            if sys.platform == "darwin":  # macOS
                os.system(f"open '{image}'")
            elif sys.platform == "win32":  # Windows
                os.system(f"start '{image}'")
            else:  # Linux
                os.system(f"xdg-open '{image}'")
        except Exception as e:
            print(f"❌ Erro ao abrir {image.name}: {e}")
    
    print(f"✅ {len(images)} imagens abertas para visualização")

def show_file_paths():
    """
    Show file paths for manual opening
    """
    images = list_generated_images()
    
    if not images:
        return
    
    print("📁 CAMINHOS COMPLETOS DAS IMAGENS:")
    print("=" * 60)
    
    for image in images:
        abs_path = image.resolve()
        print(f"📸 {abs_path}")
    
    print()
    print("💡 Você pode abrir essas imagens manualmente em qualquer visualizador de imagens")

def main():
    """
    Main function
    """
    print("🎯 VISUALIZADOR DE ANÁLISE VISUAL")
    print("=" * 60)
    
    # Show analysis summary
    show_analysis_summary()
    
    # List images
    images = list_generated_images()
    
    if not images:
        print("⚠️ Execute primeiro: python tools/visual_analysis_generator.py")
        return
    
    # Show options
    print("🔧 OPÇÕES DE VISUALIZAÇÃO:")
    print("=" * 60)
    print("1. 🚀 Abrir todas as imagens no visualizador padrão")
    print("2. 📁 Mostrar caminhos para abertura manual")
    print("3. ❌ Sair")
    print()
    
    try:
        choice = input("Escolha uma opção (1-3): ").strip()
        
        if choice == "1":
            open_images_in_default_viewer()
        elif choice == "2":
            show_file_paths()
        elif choice == "3":
            print("👋 Saindo...")
        else:
            print("⚠️ Opção inválida")
            
    except KeyboardInterrupt:
        print("\n👋 Saindo...")
    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    main()
