#!/usr/bin/env python3
"""
Quality Analysis Tool for Photo Culling System
Ferramenta para análise de qualidade em lote com detecção de blur
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.image_quality_analyzer import ImageQualityAnalyzer
import argparse
import json
from datetime import datetime

def analyze_collection():
    """Analisa toda a coleção de imagens do sistema"""
    print("📊 ANÁLISE DE QUALIDADE - COLEÇÃO COMPLETA")
    print("="*60)
    
    input_dir = "../data/input"
    if not os.path.exists(input_dir):
        print(f"❌ Diretório de imagens não encontrado: {input_dir}")
        return
    
    # Inicializa analisador com diferentes thresholds para comparação
    thresholds = [75, 100, 150]
    
    print(f"🔍 Analisando imagens em: {input_dir}")
    print(f"🎯 Testando thresholds: {thresholds}")
    print("-" * 60)
    
    results_by_threshold = {}
    
    for threshold in thresholds:
        print(f"\n📏 THRESHOLD: {threshold}")
        print("-" * 30)
        
        analyzer = ImageQualityAnalyzer(
            blur_threshold=threshold,
            results_db=f"../data/quality/quality_analysis_t{threshold}.db"
        )
        
        # Executa análise
        stats = analyzer.analyze_folder(input_dir)
        results_by_threshold[threshold] = stats
        
        if stats['total_images'] == 0:
            print("⚠️ Nenhuma imagem encontrada")
            continue
        
        print(f"   Total: {stats['total_images']} imagens")
        print(f"   Analisadas: {stats['analyzed']}")
        print(f"   Borradas: {stats['blurry_images']} ({stats.get('blur_percentage', 0):.1f}%)")
        print(f"   Nítidas: {stats['sharp_images']}")
        
        if 'avg_blur_score' in stats:
            print(f"   Score médio: {stats['avg_blur_score']:.2f}")
            print(f"   Range: {stats['min_blur_score']:.1f} - {stats['max_blur_score']:.1f}")
        
        # Distribuição de qualidade
        quality_dist = stats.get('quality_distribution', {})
        if any(quality_dist.values()):
            print(f"   Qualidade - EXCELLENT: {quality_dist.get('EXCELLENT', 0)}, "
                  f"GOOD: {quality_dist.get('GOOD', 0)}, "
                  f"FAIR: {quality_dist.get('FAIR', 0)}, "
                  f"POOR: {quality_dist.get('POOR', 0)}")
    
    # Resumo comparativo
    print(f"\n📈 COMPARAÇÃO DE THRESHOLDS")
    print("="*60)
    print(f"{'Threshold':<10} {'Borradas':<10} {'%':<8} {'Score Médio':<12}")
    print("-" * 45)
    
    for threshold, stats in results_by_threshold.items():
        if stats['total_images'] > 0:
            blur_pct = stats.get('blur_percentage', 0)
            avg_score = stats.get('avg_blur_score', 0)
            print(f"{threshold:<10} {stats['blurry_images']:<10} {blur_pct:<7.1f}% {avg_score:<12.2f}")
    
    # Recomendações
    print(f"\n💡 RECOMENDAÇÕES:")
    print(f"   • Threshold 75: Detecta apenas casos extremamente borrados")
    print(f"   • Threshold 100: Balanceado - recomendado para uso geral")
    print(f"   • Threshold 150: Mais rigoroso - ideal para impressão/qualidade alta")
    
    # Salva relatório JSON
    report_file = f"../data/quality/quality_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump({
            'analysis_timestamp': datetime.now().isoformat(),
            'input_directory': input_dir,
            'thresholds_tested': thresholds,
            'results_by_threshold': results_by_threshold
        }, f, indent=2, default=str)
    
    print(f"\n💾 Relatório salvo em: {report_file}")

def find_problematic_images():
    """Encontra imagens com problemas de qualidade"""
    print("🔍 IDENTIFICANDO IMAGENS PROBLEMÁTICAS")
    print("="*50)
    
    analyzer = ImageQualityAnalyzer()
    
    # Busca imagens borradas
    blurry_images = analyzer.get_blurry_images(limit=20)
    
    if not blurry_images:
        print("🎉 Nenhuma imagem borrada encontrada no banco!")
        print("💡 Execute análise primeiro: python quality_analyzer.py --analyze")
        return
    
    print(f"🌫️ IMAGENS BORRADAS DETECTADAS ({len(blurry_images)} total, mostrando top 20):")
    print("-" * 50)
    
    # Categoriza por nível de problema
    extremely_blurry = []
    very_blurry = []
    moderately_blurry = []
    
    for img in blurry_images:
        score = img['blur_score']
        if score < 20:
            extremely_blurry.append(img)
        elif score < 50:
            very_blurry.append(img)
        else:
            moderately_blurry.append(img)
    
    if extremely_blurry:
        print(f"\n❌ EXTREMAMENTE BORRADAS ({len(extremely_blurry)}) - RECOMENDA-SE DESCARTAR:")
        for img in extremely_blurry[:10]:
            print(f"   📷 {img['filename']} (score: {img['blur_score']:.2f})")
    
    if very_blurry:
        print(f"\n⚠️ MUITO BORRADAS ({len(very_blurry)}) - QUALIDADE COMPROMETIDA:")
        for img in very_blurry[:10]:
            print(f"   📷 {img['filename']} (score: {img['blur_score']:.2f})")
    
    if moderately_blurry:
        print(f"\n🔸 MODERADAMENTE BORRADAS ({len(moderately_blurry)}) - AVALIAR CONTEXTO:")
        for img in moderately_blurry[:10]:
            print(f"   📷 {img['filename']} (score: {img['blur_score']:.2f})")
    
    # Estatísticas
    total_problematic = len(extremely_blurry) + len(very_blurry)
    print(f"\n📊 RESUMO:")
    print(f"   🔴 Críticas (descartar): {len(extremely_blurry)}")
    print(f"   🟡 Problemáticas: {len(very_blurry)}")
    print(f"   🟢 Aceitáveis: {len(moderately_blurry)}")
    print(f"   💯 Total problemáticas: {total_problematic}")

def generate_cleanup_suggestions():
    """Gera sugestões de limpeza baseadas na análise de qualidade"""
    print("🧹 SUGESTÕES DE LIMPEZA - BASEADO EM QUALIDADE")
    print("="*55)
    
    analyzer = ImageQualityAnalyzer()
    report = analyzer.get_quality_report()
    
    if report['summary']['total_analyzed'] == 0:
        print("❌ Nenhuma análise encontrada.")
        print("💡 Execute: python quality_analyzer.py --analyze")
        return
    
    summary = report['summary']
    
    print(f"📊 ANÁLISE ATUAL:")
    print(f"   Total analisadas: {summary['total_analyzed']}")
    print(f"   Borradas: {summary['total_blurry']} ({summary['blur_percentage']:.1f}%)")
    print(f"   Nítidas: {summary['total_sharp']}")
    
    # Cálculo de economia de espaço potencial
    if summary['total_blurry'] > 0:
        # Estimativa média de 3MB por imagem (aproximação para fotos de celular)
        avg_size_mb = 3.0
        potential_savings_mb = summary['total_blurry'] * avg_size_mb
        potential_savings_gb = potential_savings_mb / 1024
        
        print(f"\n💾 ECONOMIA POTENCIAL DE ESPAÇO:")
        print(f"   Removendo borradas: ~{potential_savings_mb:.0f} MB ({potential_savings_gb:.1f} GB)")
        print(f"   Percentual da coleção: {summary['blur_percentage']:.1f}%")
    
    # Sugestões específicas
    print(f"\n🎯 SUGESTÕES DE AÇÃO:")
    
    extremely_blurry = [img for img in analyzer.get_blurry_images() if img['blur_score'] < 20]
    very_blurry = [img for img in analyzer.get_blurry_images() if 20 <= img['blur_score'] < 50]
    
    if extremely_blurry:
        print(f"   1. 🗑️ REMOÇÃO IMEDIATA ({len(extremely_blurry)} imagens):")
        print(f"      Imagens com score < 20 são praticamente inutilizáveis")
        print(f"      Economia: ~{len(extremely_blurry) * 3:.0f} MB")
    
    if very_blurry:
        print(f"   2. ⚠️ REVISÃO MANUAL ({len(very_blurry)} imagens):")
        print(f"      Imagens com score 20-50 podem ter valor contextual")
        print(f"      Avalie se têm importância histórica/sentimental")
    
    if summary['blur_percentage'] > 20:
        print(f"   3. 📸 REVISÃO DE TÉCNICA:")
        print(f"      {summary['blur_percentage']:.1f}% de borradas indica possível problema na captura")
        print(f"      Considere verificar estabilização da câmera")
    
    # Gera script de limpeza
    if extremely_blurry:
        cleanup_script = "../data/quality/cleanup_extremely_blurry.sh"
        with open(cleanup_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Script para remover imagens extremamente borradas\n")
            f.write("# CUIDADO: Este script remove arquivos definitivamente!\n\n")
            f.write("echo 'Removendo imagens extremamente borradas...'\n")
            f.write("read -p 'Tem certeza? (y/N): ' confirm\n")
            f.write("if [[ $confirm == [yY] ]]; then\n")
            
            for img in extremely_blurry:
                f.write(f"  echo 'Removendo: {img['filename']}'\n")
                f.write(f"  # rm '../input/{img['filename']}'\n")
            
            f.write("  echo 'Limpeza concluída.'\n")
            f.write("else\n")
            f.write("  echo 'Operação cancelada.'\n")
            f.write("fi\n")
        
        os.chmod(cleanup_script, 0o755)
        print(f"\n📝 Script de limpeza gerado: {cleanup_script}")
        print(f"   ⚠️ CUIDADO: Remova o '#' antes de 'rm' para executar")

def main():
    parser = argparse.ArgumentParser(description='Quality Analysis Tool for Photo Culling')
    parser.add_argument('--analyze', '-a', action='store_true',
                       help='Analyze entire image collection with multiple thresholds')
    parser.add_argument('--problems', '-p', action='store_true',
                       help='Find problematic (blurry) images')
    parser.add_argument('--cleanup', '-c', action='store_true',
                       help='Generate cleanup suggestions based on quality')
    parser.add_argument('--report', '-r', action='store_true',
                       help='Generate detailed quality report')
    
    args = parser.parse_args()
    
    try:
        if args.analyze:
            analyze_collection()
        elif args.problems:
            find_problematic_images()
        elif args.cleanup:
            generate_cleanup_suggestions()
        elif args.report:
            analyzer = ImageQualityAnalyzer()
            report = analyzer.get_quality_report()
            print(json.dumps(report, indent=2, default=str))
        else:
            print("🔍 QUALITY ANALYZER - PHOTO CULLING SYSTEM")
            print("="*50)
            print("Análise de qualidade com detecção de blur/desfoque")
            print()
            print("Opções disponíveis:")
            print("  --analyze   (-a) : Analisa coleção completa")
            print("  --problems  (-p) : Encontra imagens problemáticas")
            print("  --cleanup   (-c) : Gera sugestões de limpeza")
            print("  --report    (-r) : Relatório detalhado (JSON)")
            print()
            print("Exemplos:")
            print("  python quality_analyzer.py --analyze")
            print("  python quality_analyzer.py --problems")
            print("  python quality_analyzer.py --cleanup")
            print()
            print("💡 Baseado no método Variance of Laplacian")
            print("💡 Threshold padrão: 100.0 (configurável)")
            
    except KeyboardInterrupt:
        print("\n❌ Operação cancelada pelo usuário")
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
