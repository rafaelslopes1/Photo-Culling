#!/usr/bin/env python3
"""
Project Cleanup Script - Photo Culling System
Script de limpeza do projeto - identificar duplicatas e arquivos desnecessários
"""

import os
import shutil
from pathlib import Path
import hashlib
import json

def calculate_file_hash(file_path):
    """Calculate MD5 hash of file content"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except:
        return None

def find_duplicate_files():
    """Find duplicate files in the project"""
    
    project_root = Path(".")
    file_hashes = {}
    duplicates = []
    
    # Extensions to check for duplicates
    extensions = ['.py', '.md', '.json', '.txt']
    
    for ext in extensions:
        for file_path in project_root.rglob(f'*{ext}'):
            if file_path.is_file():
                file_hash = calculate_file_hash(file_path)
                if file_hash:
                    if file_hash in file_hashes:
                        duplicates.append({
                            'original': file_hashes[file_hash],
                            'duplicate': str(file_path),
                            'hash': file_hash
                        })
                    else:
                        file_hashes[file_hash] = str(file_path)
    
    return duplicates

def analyze_tools_directory():
    """Analyze tools directory for cleanup opportunities"""
    
    tools_dir = Path("tools")
    if not tools_dir.exists():
        return {}
    
    analysis = {
        'test_files': [],
        'demo_files': [],
        'debug_files': [],
        'analysis_files': [],
        'utility_files': []
    }
    
    for file_path in tools_dir.glob("*.py"):
        filename = file_path.name.lower()
        
        if any(keyword in filename for keyword in ['test', 'testing']):
            analysis['test_files'].append(str(file_path))
        elif any(keyword in filename for keyword in ['demo', 'example']):
            analysis['demo_files'].append(str(file_path))
        elif any(keyword in filename for keyword in ['debug', 'simple']):
            analysis['debug_files'].append(str(file_path))
        elif any(keyword in filename for keyword in ['analysis', 'analyzer', 'comprehensive']):
            analysis['analysis_files'].append(str(file_path))
        else:
            analysis['utility_files'].append(str(file_path))
    
    return analysis

def identify_cleanup_candidates():
    """Identify files and directories that can be cleaned up"""
    
    cleanup_candidates = {
        'duplicate_files': find_duplicate_files(),
        'tools_analysis': analyze_tools_directory(),
        'large_files': [],
        'temp_files': [],
        'cache_directories': []
    }
    
    # Find large files (> 1MB)
    for file_path in Path(".").rglob("*"):
        if file_path.is_file():
            try:
                size = file_path.stat().st_size
                if size > 1024 * 1024:  # 1MB
                    cleanup_candidates['large_files'].append({
                        'path': str(file_path),
                        'size_mb': round(size / (1024 * 1024), 2)
                    })
            except:
                pass
    
    # Find temp files and cache directories
    temp_patterns = ['*.tmp', '*.temp', '*~', '*.bak', '*.orig']
    cache_dirs = ['__pycache__', '.pytest_cache', '.mypy_cache', 'node_modules']
    
    for pattern in temp_patterns:
        for file_path in Path(".").rglob(pattern):
            cleanup_candidates['temp_files'].append(str(file_path))
    
    for cache_dir in cache_dirs:
        for dir_path in Path(".").rglob(cache_dir):
            if dir_path.is_dir():
                cleanup_candidates['cache_directories'].append(str(dir_path))
    
    return cleanup_candidates

def generate_cleanup_report():
    """Generate cleanup report"""
    
    print("🧹 ANÁLISE DE LIMPEZA DO PROJETO")
    print("=" * 50)
    
    candidates = identify_cleanup_candidates()
    
    # Duplicates
    print(f"\n📁 ARQUIVOS DUPLICADOS: {len(candidates['duplicate_files'])}")
    for dup in candidates['duplicate_files']:
        print(f"   Original: {dup['original']}")
        print(f"   Duplicata: {dup['duplicate']}")
        print(f"   Hash: {dup['hash'][:8]}...")
        print()
    
    # Tools analysis
    tools = candidates['tools_analysis']
    print(f"\n🛠️  ANÁLISE DO DIRETÓRIO TOOLS:")
    print(f"   Arquivos de teste: {len(tools['test_files'])}")
    print(f"   Arquivos de demo: {len(tools['demo_files'])}")
    print(f"   Arquivos de debug: {len(tools['debug_files'])}")
    print(f"   Arquivos de análise: {len(tools['analysis_files'])}")
    print(f"   Utilitários: {len(tools['utility_files'])}")
    
    # Large files
    print(f"\n📊 ARQUIVOS GRANDES (>1MB): {len(candidates['large_files'])}")
    for large_file in sorted(candidates['large_files'], key=lambda x: x['size_mb'], reverse=True)[:10]:
        print(f"   {large_file['path']} ({large_file['size_mb']} MB)")
    
    # Temp files
    print(f"\n🗑️  ARQUIVOS TEMPORÁRIOS: {len(candidates['temp_files'])}")
    for temp_file in candidates['temp_files'][:10]:
        print(f"   {temp_file}")
    
    # Cache directories
    print(f"\n📂 DIRETÓRIOS DE CACHE: {len(candidates['cache_directories'])}")
    for cache_dir in candidates['cache_directories']:
        print(f"   {cache_dir}")
    
    # Save detailed report
    report_file = Path("CLEANUP_ANALYSIS.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Relatório detalhado salvo em: {report_file}")
    
    return candidates

if __name__ == "__main__":
    generate_cleanup_report()
