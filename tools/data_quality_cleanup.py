#!/usr/bin/env python3
"""
Data Quality Cleanup - Photo Culling System
Remove análises antigas e arquivos temporários de qualidade
"""

import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta

class DataQualityCleanup:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.quality_dir = self.project_root / "data" / "quality"
        
    def cleanup_old_analysis_files(self, days_old: int = 30) -> int:
        """Remove arquivos de análise mais antigos que X dias"""
        removed_count = 0
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        if not self.quality_dir.exists():
            return 0
        
        print(f"🧹 Limpando análises antigas (> {days_old} dias)...")
        
        # Patterns for temporary analysis files
        temp_patterns = [
            "detailed_*.png",
            "detailed_*.csv", 
            "analysis_*.json",
            "test_*.png",
            "debug_*.txt"
        ]
        
        for pattern in temp_patterns:
            for file_path in self.quality_dir.rglob(pattern):
                try:
                    # Check file age
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_date:
                        file_path.unlink()
                        removed_count += 1
                        print(f"   ✅ {file_path.name}")
                except Exception as e:
                    print(f"   ❌ Erro ao remover {file_path}: {e}")
        
        return removed_count
    
    def cleanup_empty_analysis_dirs(self) -> int:
        """Remove diretórios de análise vazios"""
        removed_count = 0
        
        if not self.quality_dir.exists():
            return 0
        
        print(f"\n📁 Removendo diretórios vazios...")
        
        # Check subdirectories
        for subdir in self.quality_dir.iterdir():
            if subdir.is_dir():
                try:
                    # Check if directory is empty or contains only temp files
                    files = list(subdir.rglob("*"))
                    if not files or all(f.name.startswith('.') for f in files):
                        shutil.rmtree(subdir)
                        removed_count += 1
                        print(f"   ✅ {subdir.name}/")
                except Exception as e:
                    print(f"   ❌ Erro ao remover {subdir}: {e}")
        
        return removed_count
    
    def consolidate_essential_reports(self):
        """Consolida relatórios essenciais"""
        print(f"\n📊 Consolidando relatórios essenciais...")
        
        # Keep only essential reports
        essential_files = [
            "analysis_dashboard.png",
            "analysis_report.json", 
            "TESTE_ABRANGENTE_RELATORIO.md",
            "blur_config.py"
        ]
        
        # Create consolidated directory
        consolidated_dir = self.quality_dir / "essential_reports"
        consolidated_dir.mkdir(exist_ok=True)
        
        # Move essential files
        for subdir in self.quality_dir.iterdir():
            if subdir.is_dir() and subdir.name != "essential_reports":
                for essential_file in essential_files:
                    source_file = subdir / essential_file
                    if source_file.exists():
                        dest_file = consolidated_dir / f"{subdir.name}_{essential_file}"
                        try:
                            shutil.copy2(source_file, dest_file)
                            print(f"   ✅ {essential_file} → essential_reports/")
                        except Exception as e:
                            print(f"   ❌ Erro ao copiar {essential_file}: {e}")
    
    def generate_cleanup_summary(self) -> dict:
        """Gera resumo da limpeza"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "quality_dir_size_mb": 0,
            "total_files": 0,
            "total_dirs": 0
        }
        
        if self.quality_dir.exists():
            # Calculate directory size
            total_size = sum(
                f.stat().st_size for f in self.quality_dir.rglob('*') 
                if f.is_file()
            )
            summary["quality_dir_size_mb"] = round(total_size / (1024 * 1024), 2)
            
            # Count files and directories
            summary["total_files"] = len([
                f for f in self.quality_dir.rglob('*') if f.is_file()
            ])
            summary["total_dirs"] = len([
                d for d in self.quality_dir.rglob('*') if d.is_dir()
            ])
        
        return summary
    
    def run_full_cleanup(self, days_old: int = 30) -> dict:
        """Executa limpeza completa"""
        print("🧹 LIMPEZA DE DADOS DE QUALIDADE")
        print("=" * 40)
        
        # Before cleanup summary
        before_summary = self.generate_cleanup_summary()
        print(f"\n📊 ANTES DA LIMPEZA:")
        print(f"   Tamanho: {before_summary['quality_dir_size_mb']} MB")
        print(f"   Arquivos: {before_summary['total_files']}")
        print(f"   Diretórios: {before_summary['total_dirs']}")
        
        # Execute cleanup
        removed_files = self.cleanup_old_analysis_files(days_old)
        removed_dirs = self.cleanup_empty_analysis_dirs()
        
        # Consolidate reports
        if removed_files > 0 or removed_dirs > 0:
            self.consolidate_essential_reports()
        
        # After cleanup summary
        after_summary = self.generate_cleanup_summary()
        
        print(f"\n📊 DEPOIS DA LIMPEZA:")
        print(f"   Tamanho: {after_summary['quality_dir_size_mb']} MB")
        print(f"   Arquivos: {after_summary['total_files']}")
        print(f"   Diretórios: {after_summary['total_dirs']}")
        
        # Calculate savings
        size_saved = before_summary['quality_dir_size_mb'] - after_summary['quality_dir_size_mb']
        files_removed = before_summary['total_files'] - after_summary['total_files']
        
        print(f"\n💾 ECONOMIA:")
        print(f"   Espaço liberado: {size_saved:.2f} MB")
        print(f"   Arquivos removidos: {files_removed}")
        print(f"   Diretórios removidos: {removed_dirs}")
        
        # Save cleanup report
        cleanup_report = {
            "timestamp": datetime.now().isoformat(),
            "before": before_summary,
            "after": after_summary,
            "removed": {
                "files": files_removed,
                "directories": removed_dirs,
                "size_mb": size_saved
            }
        }
        
        report_path = self.project_root / "DATA_QUALITY_CLEANUP_REPORT.json"
        import json
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(cleanup_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Relatório salvo em: {report_path}")
        return cleanup_report

def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Limpeza de Dados de Qualidade - Photo Culling System'
    )
    parser.add_argument('--days', type=int, default=30,
                       help='Remover arquivos mais antigos que X dias (padrão: 30)')
    parser.add_argument('--project-root', default='.',
                       help='Diretório raiz do projeto')
    
    args = parser.parse_args()
    
    cleanup = DataQualityCleanup(args.project_root)
    cleanup.run_full_cleanup(args.days)

if __name__ == "__main__":
    main()
