#!/usr/bin/env python3
"""
Script executável para limpeza completa do projeto Photo Culling System
Remove arquivos duplicados, vazios e reorganiza estrutura do projeto
"""

import os
import shutil
import json
import sys
from pathlib import Path

def load_cleanup_analysis():
    """Load cleanup analysis results"""
    analysis_file = Path("CLEANUP_ANALYSIS.json")
    if not analysis_file.exists():
        print("❌ Arquivo CLEANUP_ANALYSIS.json não encontrado!")
        return None
    
    with open(analysis_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def remove_empty_files_in_tools():
    """Remove arquivos vazios do diretório tools"""
    tools_dir = Path("tools")
    removed_files = []
    
    for file_path in tools_dir.glob("*.py"):
        if file_path.stat().st_size == 0:
            print(f"🗑️  Removendo arquivo vazio: {file_path}")
            file_path.unlink()
            removed_files.append(str(file_path))
    
    return removed_files

def consolidate_tools_directory():
    """Consolidate and rename files in tools directory"""
    tools_dir = Path("tools")
    
    # Keep only essential files
    essential_files = {
        'health_check_complete.py': 'System health monitoring',
        'integration_test.py': 'Integration testing',
        'quality_analyzer.py': 'Quality analysis utilities', 
        'project_cleanup_analysis.py': 'Project cleanup analysis',
        'ai_prediction_tester.py': 'AI prediction testing',
        'demo_system.py': 'System demonstration'
    }
    
    # Files to remove (duplicates, deprecated, or redundant)
    files_to_remove = [
        'demo_phase25_complete.py',
        'test_overexposure_img_0001.py',
        'quiet_test_suite.py',
        'test_phase25_integration.py',
        'consolidated_test_suite.py',
        'gpu_optimized_test.py',
        'testing_suite.py',
        'comprehensive_analysis_test.py',
        'unified_test_suite.py',
        'optimized_analysis_test.py',
        'integrated_system_test.py',
        'face_detection_debug.py',
        'simple_face_debug.py',
        'simple_face_recognition_debug.py',
        'generate_final_report.py',
        'system_demo.py'
    ]
    
    removed_files = []
    for file_name in files_to_remove:
        file_path = tools_dir / file_name
        if file_path.exists():
            print(f"🗑️  Removendo arquivo redundante: {file_path}")
            file_path.unlink()
            removed_files.append(str(file_path))
    
    return removed_files

def remove_large_cache_directories():
    """Remove large cache directories from .venv"""
    venv_dir = Path(".venv")
    if not venv_dir.exists():
        return []
    
    # Don't remove .venv completely, just clean cache
    cache_patterns = [
        "**/__pycache__",
        "**/*.pyc",
        "**/*.pyo"
    ]
    
    removed_items = []
    for pattern in cache_patterns:
        for item in venv_dir.glob(pattern):
            if item.is_dir():
                print(f"🗑️  Removendo cache: {item}")
                shutil.rmtree(item, ignore_errors=True)
                removed_items.append(str(item))
            elif item.is_file():
                item.unlink()
                removed_items.append(str(item))
    
    return removed_items

def remove_duplicate_documentation():
    """Remove duplicate documentation files"""
    docs_dir = Path("docs")
    
    # Remove duplicate/deprecated documentation
    files_to_remove = [
        'PHASE2_IMPLEMENTATION_PLAN.md',
        'CLEANUP_ANALYSIS.md',
        'CALIBRATION_USER_FEEDBACK.md',
        'PHASE2_5_CRITICAL_IMPROVEMENTS.md',
        'PHASE2_5_INTEGRATION_COMPLETE.md',
        'PHASE2_COMPLETION_REPORT.md'
    ]
    
    removed_files = []
    for file_name in files_to_remove:
        file_path = docs_dir / file_name
        if file_path.exists():
            print(f"🗑️  Removendo documentação duplicada: {file_path}")
            file_path.unlink()
            removed_files.append(str(file_path))
    
    # Also remove from root
    root_cleanup_files = [
        'CLEANUP_PLAN.md',
        'CLEANUP_FINAL_REPORT.md',
        'CLEANUP_SUMMARY_REPORT.md'
    ]
    
    for file_name in root_cleanup_files:
        file_path = Path(file_name)
        if file_path.exists():
            print(f"🗑️  Removendo arquivo de limpeza: {file_path}")
            file_path.unlink()
            removed_files.append(str(file_path))
    
    return removed_files

def update_gitignore():
    """Update .gitignore to prevent versioning sensitive files"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
.venv/
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
# Images (large files)
data/input/*.jpg
data/input/*.jpeg
data/input/*.png
data/input/*.JPG
data/input/*.JPEG
data/input/*.PNG

# Database files
data/features/*.db
data/labels/*.db
data/quality/*.db

# Temporary files
*.tmp
*.temp
*.log

# Large model files
data/models/*.pkl
data/models/*.joblib
data/models/*.h5

# Analysis results
CLEANUP_ANALYSIS.json
data/quality/*.json
data/quality/*.csv

# Configuration with sensitive data
config_local.json
.env
"""
    
    gitignore_path = Path(".gitignore")
    with open(gitignore_path, 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    
    print("✅ .gitignore atualizado")
    return str(gitignore_path)

def create_cleanup_report(removed_items):
    """Create final cleanup report"""
    report = {
        "cleanup_timestamp": str(Path.cwd()),
        "removed_files": removed_items,
        "summary": {
            "total_removed": len(removed_items),
            "categories": {
                "tools_files": len([f for f in removed_items if 'tools/' in f]),
                "cache_directories": len([f for f in removed_items if '__pycache__' in f]),
                "documentation": len([f for f in removed_items if '.md' in f]),
                "other": len([f for f in removed_items if not any(x in f for x in ['tools/', '__pycache__', '.md'])])
            }
        }
    }
    
    with open("CLEANUP_EXECUTION_REPORT.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report

def main():
    """Execute complete project cleanup"""
    print("🧹 INICIANDO LIMPEZA COMPLETA DO PROJETO")
    print("=" * 50)
    
    all_removed_items = []
    
    # 1. Remove empty files in tools
    print("\n1️⃣ Removendo arquivos vazios do tools/")
    removed = remove_empty_files_in_tools()
    all_removed_items.extend(removed)
    
    # 2. Consolidate tools directory
    print("\n2️⃣ Consolidando diretório tools/")
    removed = consolidate_tools_directory()
    all_removed_items.extend(removed)
    
    # 3. Remove cache directories
    print("\n3️⃣ Limpando cache directories")
    removed = remove_large_cache_directories()
    all_removed_items.extend(removed)
    
    # 4. Remove duplicate documentation
    print("\n4️⃣ Removendo documentação duplicada")
    removed = remove_duplicate_documentation()
    all_removed_items.extend(removed)
    
    # 5. Update .gitignore
    print("\n5️⃣ Atualizando .gitignore")
    gitignore_file = update_gitignore()
    
    # 6. Create cleanup report
    print("\n6️⃣ Gerando relatório de limpeza")
    report = create_cleanup_report(all_removed_items)
    
    # Summary
    print(f"\n🎉 LIMPEZA CONCLUÍDA!")
    print(f"📊 Total de itens removidos: {len(all_removed_items)}")
    print(f"📄 Relatório salvo em: CLEANUP_EXECUTION_REPORT.json")
    
    # Next steps reminder
    print(f"\n🔄 PRÓXIMOS PASSOS:")
    print(f"1. git add .")
    print(f"2. git commit -m 'cleanup: reorganize project structure and remove duplicates'")
    print(f"3. git rm --cached data/input/*.JPG")
    print(f"4. git rm --cached data/features/*.db")
    print(f"5. git commit -m 'gitignore: remove large files from version control'")
    print(f"6. git push origin main")

if __name__ == "__main__":
    main()
