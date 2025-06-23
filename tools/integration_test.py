#!/usr/bin/env python3
"""
Full Pipeline Integration Test
Teste completo de integração do pipeline otimizado
"""

import os
import sys
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

from src.core.image_processor import ImageProcessor


def test_full_pipeline():
    """Test the complete optimized pipeline"""
    print("🔄 TESTE COMPLETO DO PIPELINE OTIMIZADO")
    print("=" * 50)
    
    # Initialize processor
    processor = ImageProcessor('config.json')
    
    print(f"📋 Configuração Ativa:")
    print(f"   • Sistema: {'OTIMIZADO' if processor.use_optimized_blur else 'LEGACY'}")
    if processor.use_optimized_blur:
        strategy = processor.config.get('processing_settings', {}).get('blur_detection_optimized', {}).get('strategy', 'balanced')
        print(f"   • Estratégia: {strategy.upper()}")
        print(f"   • Threshold: {processor.blur_threshold}")
    
    # Create test directories
    test_input = "test_pipeline_input"
    test_output = "test_pipeline_output"
    
    # Clean up previous tests
    if os.path.exists(test_input):
        shutil.rmtree(test_input)
    if os.path.exists(test_output):
        shutil.rmtree(test_output)
    
    os.makedirs(test_input, exist_ok=True)
    os.makedirs(test_output, exist_ok=True)
    
    # Copy sample images for testing
    input_dir = "data/input"
    images = []
    strategy = processor.config.get('processing_settings', {}).get('blur_detection_optimized', {}).get('strategy', 'balanced')
    
    if os.path.exists(input_dir):
        images = [f for f in os.listdir(input_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:10]  # Test with 10 images
        
        print(f"\n📁 Preparando teste com {len(images)} imagens:")
        for img in images:
            src = os.path.join(input_dir, img)
            dst = os.path.join(test_input, img)
            shutil.copy2(src, dst)
            print(f"   • {img}")
    
    if not images:
        print(f"❌ Nenhuma imagem disponível para teste")
        return
    
    print(f"\n🔍 Processando imagens...")
    
    # Process images
    try:
        results = processor.process_folder(test_input, test_output)
        
        print(f"\n📊 RESULTADOS DO PIPELINE:")
        print(f"   • Total processadas: {results.get('total', 0)}")
        print(f"   • Selecionadas: {results.get('selected', 0)}")
        print(f"   • Desfocadas: {results.get('blurry', 0)}")
        print(f"   • Pouca luz: {results.get('low_light', 0)}")
        print(f"   • Falhas: {results.get('failures', 0)}")
        
        # Show folder structure
        print(f"\n📂 Estrutura de saída:")
        for root, dirs, files in os.walk(test_output):
            level = root.replace(test_output, '').count(os.sep)
            indent = '  ' * level
            folder_name = os.path.basename(root) if level > 0 else 'test_pipeline_output'
            print(f"{indent}{folder_name}/ ({len(files)} arquivos)")
            
        # Analyze specific results with optimized system
        print(f"\n🔬 ANÁLISE DETALHADA (Sistema Otimizado):")
        blur_category_counts = {'extremely_blurry': 0, 'very_blurry': 0, 
                              'blurry': 0, 'acceptable': 0, 'sharp': 0}
        
        for img in images:
            img_path = os.path.join(test_input, img)
            if os.path.exists(img_path):
                analysis = processor.quality_analyzer.analyze_single_image(img_path)
                blur_score = analysis['blur_score']
                
                # Categorize using optimized thresholds
                if blur_score < 30:
                    category = 'extremely_blurry'
                elif blur_score < 50:
                    category = 'very_blurry'
                elif blur_score < processor.blur_threshold:
                    category = 'blurry'
                elif blur_score < 150:
                    category = 'acceptable'
                else:
                    category = 'sharp'
                
                blur_category_counts[category] += 1
                
                status = "🗑️ REMOVER" if blur_score < processor.blur_threshold else "✅ MANTER"
                print(f"   • {img}: Score={blur_score:.2f} ({category}) -> {status}")
        
        print(f"\n📈 DISTRIBUIÇÃO POR CATEGORIA:")
        for category, count in blur_category_counts.items():
            percentage = (count / len(images)) * 100 if images else 0
            print(f"   • {category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        # Calculate strategy effectiveness
        kept_count = sum(1 for img in images 
                        if processor.quality_analyzer.analyze_single_image(os.path.join(test_input, img))['blur_score'] >= processor.blur_threshold)
        removal_rate = ((len(images) - kept_count) / len(images)) * 100 if images else 0
        
        print(f"\n🎯 EFETIVIDADE DA ESTRATÉGIA '{strategy.upper()}':")
        print(f"   • Taxa de remoção: {removal_rate:.1f}%")
        print(f"   • Taxa de retenção: {100-removal_rate:.1f}%")
        print(f"   • Threshold usado: {processor.blur_threshold}")
        
    except Exception as e:
        print(f"❌ Erro durante processamento: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up test directories
        if os.path.exists(test_input):
            shutil.rmtree(test_input)
        if os.path.exists(test_output):
            shutil.rmtree(test_output)
        print(f"\n🧹 Diretórios de teste removidos")
        
    print(f"\n" + "=" * 50)
    print("✅ TESTE COMPLETO DO PIPELINE CONCLUÍDO!")


def generate_integration_summary():
    """Generate integration summary report"""
    print(f"\n📋 RELATÓRIO DE INTEGRAÇÃO - SISTEMA OTIMIZADO DE BLUR DETECTION")
    print("=" * 70)
    
    processor = ImageProcessor('config.json')
    
    print(f"🔧 CONFIGURAÇÃO ATIVA:")
    print(f"   • Arquivo de config: config.json")
    print(f"   • Sistema otimizado: {'✅ ATIVO' if processor.use_optimized_blur else '❌ INATIVO'}")
    
    if processor.use_optimized_blur:
        config = processor.config.get('processing_settings', {}).get('blur_detection_optimized', {})
        strategy = config.get('strategy', 'balanced')
        strategies = config.get('strategies', {})
        
        print(f"   • Estratégia atual: {strategy.upper()}")
        print(f"   • Threshold ativo: {processor.blur_threshold}")
        
        print(f"\n🎚️ ESTRATÉGIAS DISPONÍVEIS:")
        for strat_name, strat_config in strategies.items():
            threshold = strat_config.get('threshold', 'N/A')
            desc = strat_config.get('description', 'Sem descrição')
            current = " (ATUAL)" if strat_name == strategy else ""
            print(f"   • {strat_name.upper()}{current}: {threshold} - {desc}")
        
        print(f"\n📊 VALIDAÇÃO SUPERVISIONADA:")
        validation_config = config.get('supervised_validation', {})
        print(f"   • Habilitada: {'✅' if validation_config.get('enabled') else '❌'}")
        print(f"   • Base de dados: {validation_config.get('database_path', 'N/A')}")
        print(f"   • Confiança mínima: {validation_config.get('min_confidence', 'N/A')}")
    
    print(f"\n🚀 COMO USAR:")
    print(f"   1. python main.py --classify --input-dir data/input")
    print(f"   2. python demo_integrated_system.py")
    print(f"   3. Ajustar estratégia em config.json se necessário")
    
    print(f"\n📚 DOCUMENTAÇÃO:")
    print(f"   • docs/BLUR_DETECTION.md - Documentação técnica")
    print(f"   • docs/BLUR_ANALYSIS_EXECUTIVE_SUMMARY.md - Resumo executivo")
    print(f"   • tools/ - Scripts de análise e validação")
    
    print(f"\n🎯 RECOMENDAÇÕES:")
    print(f"   • Para uso geral: strategy='balanced' (threshold=78)")
    print(f"   • Para arquivo pessoal: strategy='conservative' (threshold=50)")
    print(f"   • Para portfólio profissional: strategy='aggressive' (threshold=145)")
    
    print("=" * 70)
    print("✅ SISTEMA OTIMIZADO DE BLUR DETECTION - INTEGRAÇÃO COMPLETA!")


def main():
    """Main test function"""
    print("🧪 TESTE FINAL DE INTEGRAÇÃO - BLUR DETECTION OTIMIZADO")
    print("Sistema de Photo Culling com Detecção Avançada de Desfoque")
    print("=" * 70)
    
    try:
        # Run full pipeline test
        test_full_pipeline()
        
        # Generate integration summary
        generate_integration_summary()
        
    except Exception as e:
        print(f"❌ Erro durante teste: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
