#!/usr/bin/env python3
"""
System Health Check - Photo Culling v2.0
Verificação de integridade do sistema com blur detection otimizado
"""

import os
import sys
import json
from pathlib import Path
import importlib.util

def check_system_health():
    """Comprehensive system health check"""
    print("🔍 VERIFICAÇÃO DE INTEGRIDADE DO SISTEMA")
    print("Photo Culling v2.0 - Blur Detection Otimizado")
    print("=" * 60)
    
    health_report = {
        'configuration': False,
        'modules': False,
        'directories': False,
        'files': False,
        'integration': False,
        'data': False
    }
    
    # Check configuration
    print("\n🔧 Verificando configuração...")
    try:
        if os.path.exists('config.json'):
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            blur_config = config.get('processing_settings', {}).get('blur_detection_optimized', {})
            enabled = blur_config.get('enabled', False)
            strategy = blur_config.get('strategy', 'balanced')
            threshold = 78 if strategy == 'balanced' else 50
            
            print(f"   ✅ config.json encontrado")
            print(f"   ✅ Blur detection otimizado: {enabled}")
            print(f"   ✅ Estratégia: {strategy}")
            print(f"   ✅ Threshold estimado: {threshold}")
            health_report['configuration'] = True
        else:
            print(f"   ❌ config.json não encontrado")
    except Exception as e:
        print(f"   ❌ Erro na configuração: {e}")
    
    # Check Python modules
    print(f"\n📦 Verificando módulos Python...")
    required_modules = [
        'src.core.image_processor',
        'src.core.image_quality_analyzer'
    ]
    
    sys.path.insert(0, str(Path.cwd()))
    sys.path.insert(0, str(Path.cwd() / 'src'))
    
    modules_ok = True
    for module in required_modules:
        try:
            spec = importlib.util.find_spec(module)
            if spec:
                print(f"   ✅ {module}")
            else:
                print(f"   ❌ {module} - não encontrado")
                modules_ok = False
        except Exception as e:
            print(f"   ❌ {module} - erro: {e}")
            modules_ok = False
    
    health_report['modules'] = modules_ok
    
    # Check directories
    print(f"\n📁 Verificando diretórios...")
    required_dirs = [
        'src/core',
        'data/quality', 
        'data/labels',
        'docs',
        'tools'
    ]
    
    dirs_ok = True
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"   ✅ {directory}")
        else:
            print(f"   ❌ {directory} - não encontrado")
            dirs_ok = False
    
    health_report['directories'] = dirs_ok
    
    # Check key files
    print(f"\n📄 Verificando arquivos principais...")
    key_files = [
        'config.json',
        'main.py',
        'src/core/image_processor.py',
        'src/core/image_quality_analyzer.py',
        'data/quality/blur_config.py',
        'docs/BLUR_DETECTION.md',
        'docs/BLUR_ANALYSIS_EXECUTIVE_SUMMARY.md',
        'docs/INTEGRATION_STATUS_FINAL.md'
    ]
    
    files_ok = True
    for file_path in key_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - não encontrado")
            files_ok = False
    
    health_report['files'] = files_ok
    
    # Test integration
    print(f"\n🧪 Testando integração...")
    try:
        from src.core.image_processor import ImageProcessor
        
        processor = ImageProcessor('config.json')
        print(f"   ✅ ImageProcessor inicializado")
        print(f"   ✅ Blur detection otimizado: {processor.use_optimized_blur}")
        print(f"   ✅ Threshold ativo: {processor.blur_threshold}")
        print(f"   ✅ Quality analyzer disponível: {hasattr(processor, 'quality_analyzer')}")
        health_report['integration'] = True
        
    except Exception as e:
        print(f"   ❌ Erro na integração: {e}")
    
    # Check sample data
    print(f"\n🖼️  Verificando dados de amostra...")
    try:
        input_dir = "data/input"
        if os.path.exists(input_dir):
            images = [f for f in os.listdir(input_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_count = len(images)
            print(f"   ✅ Diretório de entrada: {image_count} imagens")
            if image_count >= 5:
                print(f"   ✅ Dados suficientes para teste")
                health_report['data'] = True
            else:
                print(f"   ⚠️  Poucas imagens para teste completo")
        else:
            print(f"   ❌ Diretório data/input não encontrado")
    except Exception as e:
        print(f"   ❌ Erro verificando dados: {e}")
    
    # Generate report
    print(f"\n" + "=" * 60)
    print(f"📊 RESUMO DA VERIFICAÇÃO:")
    
    status_icon = lambda status: "✅ OK    " if status else "❌ FALHA"
    
    print(f"   {status_icon(health_report['configuration'])} Configuração")
    print(f"   {status_icon(health_report['modules'])} Módulos Python")
    print(f"   {status_icon(health_report['directories'])} Diretórios")
    print(f"   {status_icon(health_report['files'])} Arquivos principais")
    print(f"   {status_icon(health_report['integration'])} Integração")
    print(f"   {status_icon(health_report['data'])} Dados de amostra")
    
    overall_health = all(health_report.values())
    
    print(f"\n" + "=" * 60)
    if overall_health:
        print(f"🎉 SISTEMA COMPLETAMENTE OPERACIONAL!")
        print(f"\n🚀 Próximos passos:")
        print(f"   • python main.py --classify --input-dir data/input")
        print(f"   • python demo_integrated_system.py")
        print(f"   • python main.py --web-interface")
        
        print(f"\n📚 Documentação:")
        print(f"   • docs/INTEGRATION_STATUS_FINAL.md - Status completo")
        print(f"   • docs/BLUR_DETECTION.md - Documentação técnica")
        print(f"   • docs/BLUR_ANALYSIS_EXECUTIVE_SUMMARY.md - Resumo executivo")
    else:
        print(f"⚠️  SISTEMA COM PROBLEMAS - Verifique os itens marcados com ❌")
        print(f"\nPara resolver problemas:")
        print(f"   • Instale dependências: pip install -r requirements.txt")
        print(f"   • Verifique estrutura de arquivos")
        print(f"   • Consulte documentação em docs/")
    
    return overall_health


def main():
    """Main health check function"""
    try:
        is_healthy = check_system_health()
        sys.exit(0 if is_healthy else 1)
    except Exception as e:
        print(f"❌ Erro crítico durante verificação: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
