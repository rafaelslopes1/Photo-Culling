#!/usr/bin/env python3
"""
Integrated Blur Detection System Demo
Demonstração do sistema integrado de detecção de blur
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.core.image_processor import ImageProcessor


def test_integrated_system():
    """Test the integrated blur detection system"""
    print("🔍 Testando Sistema Integrado de Detecção de Blur")
    print("=" * 60)
    
    # Test configuration loading
    processor = ImageProcessor("config.json")
    
    print(f"📋 Configuração carregada:")
    print(f"   • Blur detection otimizado: {processor.use_optimized_blur}")
    if processor.use_optimized_blur:
        strategy = processor.config.get('processing_settings', {}).get('blur_detection_optimized', {}).get('strategy', 'balanced')
        print(f"   • Estratégia: {strategy}")
        print(f"   • Threshold: {processor.blur_threshold}")
    else:
        print(f"   • Threshold legacy: {processor.config['processing_settings']['blur_threshold']}")
    
    print("\n" + "=" * 60)
    
    # Test single image analysis
    input_dir = "data/input"
    if os.path.exists(input_dir):
        images = [f for f in os.listdir(input_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if images:
            print(f"🖼️  Testando com imagens do diretório {input_dir}")
            print(f"   • Total de imagens: {len(images)}")
            
            # Test first few images
            test_images = images[:5]
            print(f"   • Testando primeiras {len(test_images)} imagens:")
            
            for img_name in test_images:
                img_path = os.path.join(input_dir, img_name)
                
                if processor.use_optimized_blur:
                    result = processor.quality_analyzer.analyze_single_image(img_path)
                    blur_score = result['blur_score']
                    is_blurry = blur_score < processor.blur_threshold
                    category = "DESFOCADA" if is_blurry else "NÍTIDA"
                    print(f"     - {img_name}: Score={blur_score:.2f}, {category}")
                else:
                    print(f"     - {img_name}: Usando detecção legacy")
        else:
            print(f"❌ Nenhuma imagem encontrada em {input_dir}")
    else:
        print(f"❌ Diretório {input_dir} não encontrado")
    
    print("\n" + "=" * 60)
    
    # Show configuration recommendations
    print("📈 Recomendações de Configuração:")
    strategies = processor.config.get('processing_settings', {}).get('blur_detection_optimized', {}).get('strategies', {})
    
    for strategy_name, strategy_config in strategies.items():
        threshold = strategy_config.get('threshold', 'N/A')
        description = strategy_config.get('description', 'Sem descrição')
        current = " (ATUAL)" if strategy_name == processor.config.get('processing_settings', {}).get('blur_detection_optimized', {}).get('strategy') else ""
        print(f"   • {strategy_name.upper()}{current}: {threshold} - {description}")
    
    print("\n" + "=" * 60)
    print("✅ Teste do sistema integrado concluído!")
    
    return processor


def demonstrate_strategy_comparison():
    """Demonstrate different threshold strategies"""
    print("\n🔄 Comparação de Estratégias de Threshold")
    print("=" * 60)
    
    input_dir = "data/input"
    if not os.path.exists(input_dir):
        print(f"❌ Diretório {input_dir} não encontrado")
        return
    
    images = [f for f in os.listdir(input_dir) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]  # Test with 3 images
    
    if not images:
        print(f"❌ Nenhuma imagem encontrada em {input_dir}")
        return
    
    strategies = ['conservative', 'balanced', 'aggressive', 'very_aggressive']
    
    from src.core.image_quality_analyzer import ImageQualityAnalyzer
    analyzer = ImageQualityAnalyzer()
    
    print(f"🖼️  Testando com {len(images)} imagens:")
    
    for img_name in images:
        img_path = os.path.join(input_dir, img_name)
        result = analyzer.analyze_single_image(img_path)
        blur_score = result['blur_score']
        
        print(f"\n   📷 {img_name} (Score: {blur_score:.2f})")
        
        for strategy in strategies:
            # Create temporary processor with this strategy
            temp_config = {
                "processing_settings": {
                    "blur_detection_optimized": {
                        "enabled": True,
                        "strategy": strategy,
                        "strategies": {
                            "conservative": {"threshold": 50},
                            "balanced": {"threshold": 78},
                            "aggressive": {"threshold": 145},
                            "very_aggressive": {"threshold": 98}
                        }
                    }
                }
            }
            
            # Save temp config
            with open("temp_config.json", "w") as f:
                json.dump(temp_config, f)
            
            temp_processor = ImageProcessor("temp_config.json")
            is_blurry = blur_score < temp_processor.blur_threshold
            category = "DESFOCADA" if is_blurry else "NÍTIDA"
            
            print(f"      {strategy.upper():>15}: Threshold={temp_processor.blur_threshold:>3} -> {category}")
        
        # Clean up
        if os.path.exists("temp_config.json"):
            os.remove("temp_config.json")
    
    print("\n" + "=" * 60)
    print("✅ Comparação de estratégias concluída!")


def main():
    """Main demo function"""
    print("🚀 Demonstração do Sistema Integrado de Blur Detection")
    print("Sistema de Photo Culling com Detecção Otimizada de Desfoque")
    print("=" * 60)
    
    try:
        # Test integrated system
        processor = test_integrated_system()
        
        # Demonstrate strategy comparison
        demonstrate_strategy_comparison()
        
        print(f"\n📄 Para mais informações, consulte:")
        print(f"   • docs/BLUR_DETECTION.md")
        print(f"   • docs/BLUR_ANALYSIS_EXECUTIVE_SUMMARY.md")
        
    except Exception as e:
        print(f"❌ Erro durante demonstração: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
