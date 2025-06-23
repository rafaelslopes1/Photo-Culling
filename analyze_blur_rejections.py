#!/usr/bin/env python3
"""
Análise Específica para Otimização de Blur Detection
Foca em imagens rejeitadas por blur nos dados rotulados
"""

import sys
import sqlite3
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.core.image_quality_analyzer import ImageQualityAnalyzer


def analyze_blur_rejections():
    """Analisa especificamente rejeições por blur nos dados rotulados"""
    print("🔍 ANÁLISE ESPECÍFICA - REJEIÇÕES POR BLUR")
    print("=" * 60)
    
    # Connect to labels database
    db_path = "data/labels/labels.db"
    if not Path(db_path).exists():
        print(f"❌ Database não encontrado: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all rejection images with reasons
    cursor.execute("""
        SELECT filename, label, rejection_reason, confidence 
        FROM labels 
        WHERE label = 'rejection' 
        AND rejection_reason IS NOT NULL
        ORDER BY rejection_reason
    """)
    
    rejections = cursor.fetchall()
    print(f"📊 Total de rejeições com motivo: {len(rejections)}")
    
    # Categorize by rejection reason
    rejection_reasons = {}
    for filename, label, reason, confidence in rejections:
        if reason not in rejection_reasons:
            rejection_reasons[reason] = []
        rejection_reasons[reason].append({
            'filename': filename,
            'confidence': confidence
        })
    
    print(f"\n📋 DISTRIBUIÇÃO DE MOTIVOS DE REJEIÇÃO:")
    for reason, items in rejection_reasons.items():
        print(f"   • {reason}: {len(items)} imagens")
    
    # Focus on blur rejections
    blur_rejections = rejection_reasons.get('blur', [])
    if not blur_rejections:
        print(f"\n❌ Nenhuma rejeição específica por 'blur' encontrada")
        print(f"   Motivos disponíveis: {list(rejection_reasons.keys())}")
        conn.close()
        return
    
    print(f"\n🎯 ANÁLISE ESPECÍFICA - REJEIÇÕES POR BLUR ({len(blur_rejections)} imagens)")
    print("-" * 50)
    
    # Analyze blur scores for these specific rejections
    analyzer = ImageQualityAnalyzer()
    blur_scores = []
    analyzed_count = 0
    
    for item in blur_rejections:
        filename = item['filename']
        confidence = item['confidence']
        
        # Try to find the image file
        possible_paths = [
            f"data/input/{filename}",
            f"input/{filename}"
        ]
        
        image_path = None
        for path in possible_paths:
            if Path(path).exists():
                image_path = path
                break
        
        if image_path:
            try:
                result = analyzer.analyze_single_image(image_path)
                blur_score = result['blur_score']
                blur_scores.append(blur_score)
                analyzed_count += 1
                
                print(f"   • {filename}: Score={blur_score:.2f} (confiança={confidence:.2f if confidence else 'N/A'})")
            except Exception as e:
                print(f"   ⚠️  {filename}: Erro na análise - {e}")
        else:
            print(f"   ❌ {filename}: Arquivo não encontrado")
    
    if blur_scores:
        avg_blur = sum(blur_scores) / len(blur_scores)
        min_blur = min(blur_scores)
        max_blur = max(blur_scores)
        
        print(f"\n📈 ESTATÍSTICAS DAS REJEIÇÕES POR BLUR:")
        print(f"   • Imagens analisadas: {analyzed_count}/{len(blur_rejections)}")
        print(f"   • Blur score médio: {avg_blur:.2f}")
        print(f"   • Blur score mínimo: {min_blur:.2f}")
        print(f"   • Blur score máximo: {max_blur:.2f}")
        
        # Compare with current thresholds
        print(f"\n🎚️ COMPARAÇÃO COM THRESHOLDS ATUAIS:")
        thresholds = {
            'conservative': 50,
            'balanced': 78,
            'aggressive': 145,
            'very_aggressive': 98
        }
        
        for name, threshold in thresholds.items():
            would_detect = sum(1 for score in blur_scores if score < threshold)
            detection_rate = (would_detect / len(blur_scores)) * 100
            print(f"   • {name.upper()} ({threshold}): detectaria {would_detect}/{len(blur_scores)} ({detection_rate:.1f}%)")
        
        # Recommend optimal threshold
        # Sort blur scores to find a good threshold
        sorted_scores = sorted(blur_scores)
        
        # Try to find threshold that catches 80% of blur rejections
        target_detection = 0.8
        target_index = int(len(sorted_scores) * target_detection)
        if target_index < len(sorted_scores):
            optimal_threshold = sorted_scores[target_index]
            print(f"\n💡 THRESHOLD OTIMIZADO SUGERIDO:")
            print(f"   • Para detectar {target_detection*100:.0f}% das suas rejeições por blur: {optimal_threshold:.0f}")
            
            # Test this threshold
            detected = sum(1 for score in blur_scores if score < optimal_threshold)
            detection_rate = (detected / len(blur_scores)) * 100
            print(f"   • Taxa de detecção: {detected}/{len(blur_scores)} ({detection_rate:.1f}%)")
    
    conn.close()
    print(f"\n" + "=" * 60)
    print("✅ Análise específica de blur concluída!")


def recommend_custom_threshold():
    """Recomenda threshold customizado baseado nos dados"""
    print(f"\n🔧 COMO CONFIGURAR THRESHOLD CUSTOMIZADO:")
    print("=" * 50)
    
    print(f"1. Edite o arquivo config.json:")
    print(f"   \"blur_detection_optimized\": {{")
    print(f"     \"enabled\": true,")
    print(f"     \"strategy\": \"custom\",")
    print(f"     \"custom_threshold\": SEU_VALOR_AQUI")
    print(f"   }}")
    
    print(f"\n2. Ou use um dos scripts de teste:")
    print(f"   python tools/blur_threshold_supervised_eval.py")
    print(f"   python tools/blur_analysis_detailed.py")
    
    print(f"\n3. Para testar diferentes valores:")
    print(f"   python demo_integrated_system.py")


def main():
    """Main analysis function"""
    try:
        analyze_blur_rejections()
        recommend_custom_threshold()
    except Exception as e:
        print(f"❌ Erro durante análise: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
