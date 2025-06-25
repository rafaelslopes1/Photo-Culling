#!/usr/bin/env python3
"""
View Quick Fix Results - Visualizador de Resultados de Correção Rápida
"""

import os
import cv2
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def view_quick_fix_results():
    """View the quick fix results"""
    results_dir = Path("data/analysis_results/quick_fix")
    
    if not results_dir.exists():
        logger.error("❌ Diretório de resultados não encontrado")
        return
    
    # Load summary
    summary_path = results_dir / "quick_fix_summary.json"
    if summary_path.exists():
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        logger.info("📊 RESUMO DOS RESULTADOS DE CORREÇÃO RÁPIDA")
        logger.info("=" * 50)
        logger.info(f"Total de imagens processadas: {summary['total_images']}")
        logger.info(f"Faces por imagem (média): {summary['average_faces']:.1f}")
        logger.info(f"Pessoas por imagem (média): {summary['average_persons']:.1f}")
        logger.info(f"Detecção forçada bem-sucedida: {summary['forced_detection_success']}/{summary['total_images']}")
        logger.info(f"Imagens com pessoas: {summary['images_with_people']}/{summary['total_images']}")
        
        logger.info("\n📋 Distribuição de Qualidade:")
        for quality, count in summary['quality_distribution'].items():
            logger.info(f"   {quality}: {count} imagens")
    
    # Show individual results
    logger.info("\n🔍 RESULTADOS INDIVIDUAIS:")
    logger.info("=" * 50)
    
    json_files = list(results_dir.glob("*.json"))
    json_files = [f for f in json_files if f.name != "quick_fix_summary.json"]
    json_files.sort()
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        image_name = Path(result['image_path']).name
        logger.info(f"\n📸 {image_name}:")
        
        # Detection comparison
        std = result['standard_detection']
        enh = result['enhanced_detection']
        logger.info(f"   Faces: {std['faces']} (padrão) → {enh['faces']} (final)")
        logger.info(f"   Pessoas: {std['persons']} (padrão) → {enh['persons']} (final)")
        if enh['forced_persons'] > 0:
            logger.info(f"   ✅ Detecção forçada adicionou: {enh['forced_persons']} pessoas")
        
        # Metrics
        metrics = result['metrics']
        logger.info(f"   Blur Score: {metrics['blur_score']:.1f}")
        logger.info(f"   Qualidade: {metrics['quality_level']}")
        logger.info(f"   Recomendação: {metrics['recommendation']}")
        
        # Sources
        sources = result['person_detection_sources']
        logger.info(f"   Fontes de detecção: {', '.join(set(sources))}")
    
    # Show available annotated images
    logger.info("\n🖼️ IMAGENS ANOTADAS DISPONÍVEIS:")
    logger.info("=" * 50)
    
    image_files = list(results_dir.glob("*.jpg"))
    image_files.sort()
    
    for img_file in image_files:
        logger.info(f"   {img_file.name}")
    
    if image_files:
        logger.info(f"\n📁 Localização: {results_dir}")
        logger.info("💡 Dica: Abra as imagens para ver as anotações visuais com:")
        logger.info(f"   - Bounding boxes coloridos")
        logger.info(f"   - Métricas corrigidas")
        logger.info(f"   - Landmarks de pose")
        logger.info(f"   - Fontes de detecção")


if __name__ == "__main__":
    view_quick_fix_results()
