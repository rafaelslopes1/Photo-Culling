#!/usr/bin/env python3
"""
Photo Culling System - Health Check
Verificação de integridade do sistema reorganizado
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

def check_structure():
    """Verifica se a estrutura de diretórios está correta"""
    required_dirs = [
        'src',
        'src/core',
        'src/web', 
        'src/utils',
        'data',
        'data/input',
        'data/features',
        'data/labels',
        'data/models',
        'docs',
        'tools'
    ]
    
    print("🔍 Verificando estrutura de diretórios...")
    missing = []
    for directory in required_dirs:
        if not os.path.exists(directory):
            missing.append(directory)
    
    if missing:
        print(f"❌ Diretórios faltando: {missing}")
        return False
    else:
        print("✅ Estrutura de diretórios OK")
        return True

def check_core_modules():
    """Verifica se os módulos principais podem ser importados"""
    print("🔍 Verificando módulos principais...")
    modules = [
        'core.feature_extractor',
        'core.ai_classifier', 
        'core.image_processor',
        'web.app',
        'utils.config_manager',
        'utils.data_utils'
    ]
    
    failed = []
    for module in modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except Exception as e:
            print(f"  ❌ {module}: {e}")
            failed.append(module)
    
    if failed:
        print(f"❌ Módulos com problemas: {failed}")
        return False
    else:
        print("✅ Todos os módulos OK")
        return True

def check_files():
    """Verifica se arquivos essenciais existem"""
    print("🔍 Verificando arquivos essenciais...")
    required_files = [
        'main.py',
        'config.json',
        'requirements.txt',
        'README.md',
        'data/labels/labels.db',
        'data/features/features.db'
    ]
    
    missing = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing.append(file_path)
        else:
            print(f"  ✅ {file_path}")
    
    if missing:
        print(f"❌ Arquivos faltando: {missing}")
        return False
    else:
        print("✅ Todos os arquivos essenciais OK")
        return True

def check_data():
    """Verifica integridade dos dados"""
    print("🔍 Verificando dados...")
    
    # Contar imagens
    input_dir = Path('data/input')
    if input_dir.exists():
        image_count = len(list(input_dir.glob('*.JPG'))) + len(list(input_dir.glob('*.jpg')))
        print(f"  📷 Imagens encontradas: {image_count}")
    else:
        print("  ❌ Diretório de imagens não encontrado")
        return False
    
    # Verificar bancos de dados
    labels_db = Path('data/labels/labels.db')
    features_db = Path('data/features/features.db')
    
    if labels_db.exists():
        size_mb = labels_db.stat().st_size / (1024 * 1024)
        print(f"  🏷️  Base de rótulos: {size_mb:.1f} MB")
    else:
        print("  ❌ Base de rótulos não encontrada")
        return False
        
    if features_db.exists():
        size_mb = features_db.stat().st_size / (1024 * 1024)
        print(f"  🎯 Base de características: {size_mb:.1f} MB")
    else:
        print("  ❌ Base de características não encontrada")
        return False
    
    print("✅ Dados OK")
    return True

def main():
    """Executa verificação completa do sistema"""
    print("🩺 Photo Culling System - Health Check")
    print("=" * 50)
    
    checks = [
        check_structure(),
        check_core_modules(),
        check_files(),
        check_data()
    ]
    
    print("\n" + "=" * 50)
    if all(checks):
        print("🎉 SISTEMA SAUDÁVEL - Todas as verificações passaram!")
        print("✅ O Photo Culling System está pronto para uso.")
        return 0
    else:
        print("❌ PROBLEMAS DETECTADOS - Algumas verificações falharam.")
        print("⚠️  Verifique os erros acima antes de usar o sistema.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
