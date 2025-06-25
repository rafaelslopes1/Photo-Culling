#!/usr/bin/env python3
"""
Visualizador de Resultados da Demonstração
Ferramenta para exibir as visualizações anotadas geradas
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def display_annotated_images(results_dir: str = "data/analysis_results/production_showcase"):
    """
    Exibe as imagens anotadas geradas pela demonstração
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ Diretório não encontrado: {results_dir}")
        return
    
    # Find annotated images
    annotated_images = list(results_path.glob("annotated_*.JPG"))
    
    if not annotated_images:
        print(f"❌ Nenhuma imagem anotada encontrada em: {results_dir}")
        return
    
    print(f"🔍 Encontradas {len(annotated_images)} imagens anotadas")
    print("Pressione qualquer tecla para avançar, 'q' para sair")
    
    for i, img_path in enumerate(sorted(annotated_images), 1):
        print(f"\n[{i}/{len(annotated_images)}] Exibindo: {img_path.name}")
        
        # Load and display image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ Erro ao carregar: {img_path.name}")
            continue
        
        # Resize for display if too large
        height, width = img.shape[:2]
        max_display_height = 800
        
        if height > max_display_height:
            scale = max_display_height / height
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        # Display image
        cv2.imshow(f"Photo Culling Analysis - {img_path.name}", img)
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        
        if key == ord('q'):
            print("Visualização interrompida pelo usuário")
            break
    
    print("✅ Visualização concluída")

def list_analysis_files(results_dir: str = "data/analysis_results/production_showcase"):
    """
    Lista todos os arquivos gerados na demonstração
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ Diretório não encontrado: {results_dir}")
        return
    
    print(f"📁 ARQUIVOS GERADOS EM: {results_dir}")
    print("=" * 60)
    
    # JSON results
    json_files = list(results_path.glob("*.json"))
    if json_files:
        print("\n📄 RESULTADOS JSON:")
        for json_file in sorted(json_files):
            size_mb = json_file.stat().st_size / (1024 * 1024)
            print(f"  • {json_file.name} ({size_mb:.1f} MB)")
    
    # Markdown reports
    md_files = list(results_path.glob("*.md"))
    if md_files:
        print("\n📝 RELATÓRIOS:")
        for md_file in sorted(md_files):
            size_kb = md_file.stat().st_size / 1024
            print(f"  • {md_file.name} ({size_kb:.1f} KB)")
    
    # Annotated images
    img_files = list(results_path.glob("annotated_*.JPG"))
    if img_files:
        print(f"\n📊 VISUALIZAÇÕES ANOTADAS ({len(img_files)} imagens):")
        for img_file in sorted(img_files):
            size_mb = img_file.stat().st_size / (1024 * 1024)
            original_name = img_file.name.replace("annotated_", "")
            print(f"  • {original_name} → {img_file.name} ({size_mb:.1f} MB)")
    
    print(f"\n✅ Total de arquivos: {len(list(results_path.glob('*')))}")

def show_summary_statistics(results_dir: str = "data/analysis_results/production_showcase"):
    """
    Mostra estatísticas resumidas dos resultados
    """
    import json
    
    results_path = Path(results_dir)
    json_files = list(results_path.glob("production_showcase_results_*.json"))
    
    if not json_files:
        print(f"❌ Arquivo de resultados JSON não encontrado em: {results_dir}")
        return
    
    # Load latest results
    latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
    
    with open(latest_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    stats = data.get('summary_statistics', {})
    
    print("📈 ESTATÍSTICAS DA DEMONSTRAÇÃO")
    print("=" * 50)
    print(f"🔍 Total analisado: {stats.get('total_analyzed', 0)} imagens")
    print(f"⭐ Score médio: {stats.get('average_quality_score', 0):.1f}/1.0")
    print(f"🔧 Nitidez média: {stats.get('average_sharpness', 0):.1f}")
    print(f"👤 Pessoas/imagem: {stats.get('average_persons_per_image', 0):.1f}")
    
    dist = stats.get('rating_distribution', {})
    print(f"\n🏆 DISTRIBUIÇÃO DE QUALIDADE:")
    rating_emojis = {
        'excellent': '🌟',
        'good': '✅', 
        'fair': '⚖️',
        'poor': '⚠️',
        'reject': '❌'
    }
    
    total = stats.get('total_analyzed', 0)
    for rating, count in dist.items():
        if count > 0:
            pct = (count / total) * 100 if total > 0 else 0
            emoji = rating_emojis.get(rating, '❓')
            print(f"  {emoji} {rating.capitalize()}: {count} ({pct:.1f}%)")
    
    print(f"\n🥇 DESTAQUES:")
    print(f"  🏆 Melhor: {stats.get('best_image', 'N/A')}")
    print(f"  🔧 Necessita atenção: {stats.get('worst_image', 'N/A')}")

def main():
    """
    Menu principal do visualizador
    """
    print("🖼️ VISUALIZADOR DE RESULTADOS - PHOTO CULLING SYSTEM")
    print("=" * 60)
    
    while True:
        print("\nOPÇÕES:")
        print("1. 📊 Listar arquivos gerados")
        print("2. 📈 Mostrar estatísticas")
        print("3. 🖼️ Visualizar imagens anotadas")
        print("4. ❌ Sair")
        
        choice = input("\nEscolha uma opção (1-4): ").strip()
        
        if choice == '1':
            list_analysis_files()
        elif choice == '2':
            show_summary_statistics()
        elif choice == '3':
            print("\nIniciando visualização...")
            print("INSTRUÇÕES:")
            print("  • Pressione qualquer tecla para avançar")
            print("  • Pressione 'q' para sair")
            input("Pressione Enter para continuar...")
            display_annotated_images()
        elif choice == '4':
            print("👋 Saindo...")
            break
        else:
            print("❌ Opção inválida")

if __name__ == "__main__":
    main()
