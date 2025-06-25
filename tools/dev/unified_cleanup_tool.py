#!/usr/bin/env python3
"""
Unified Project Cleanup Tool - Photo Culling System v2.5
Ferramenta unificada de limpeza: análise, execução e manutenção
"""

import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

class UnifiedCleanupTool:
    """
    Ferramenta unificada que combina análise, execução e manutenção
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "analysis": {},
            "cleanup_results": {},
            "recommendations": []
        }
    
    def calculate_file_hash(self, file_path: Path) -> Optional[str]:
        """Calcula hash MD5 do arquivo"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return None
    
    def analyze_project_structure(self) -> Dict:
        """Analisa estrutura do projeto"""
        analysis = {
            "duplicate_files": [],
            "large_files": [],
            "temp_files": [],
            "cache_dirs": [],
            "empty_dirs": [],
            "redundant_reports": [],
            "old_docs": []
        }
        
        # Find duplicates
        file_hashes = {}
        extensions = ['.py', '.md', '.json', '.txt']
        
        for ext in extensions:
            for file_path in self.project_root.rglob(f'*{ext}'):
                if file_path.is_file() and '.venv' not in str(file_path):
                    file_hash = self.calculate_file_hash(file_path)
                    if file_hash:
                        if file_hash in file_hashes:
                            analysis["duplicate_files"].append({
                                'original': file_hashes[file_hash],
                                'duplicate': str(file_path),
                                'hash': file_hash
                            })
                        else:
                            file_hashes[file_hash] = str(file_path)
        
        # Find large files
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file() and '.venv' not in str(file_path):
                try:
                    size = file_path.stat().st_size
                    if size > 1024 * 1024:  # 1MB
                        analysis["large_files"].append({
                            'path': str(file_path),
                            'size_mb': round(size / (1024 * 1024), 2)
                        })
                except:
                    pass
        
        # Find temp files and cache
        temp_patterns = ['*.tmp', '*.temp', '*~', '*.bak', '*.orig', '*.log']
        cache_dirs = ['__pycache__', '.pytest_cache', '.mypy_cache']
        
        for pattern in temp_patterns:
            analysis["temp_files"].extend(
                str(p) for p in self.project_root.rglob(pattern) 
                if '.venv' not in str(p)
            )
        
        for cache_dir in cache_dirs:
            analysis["cache_dirs"].extend(
                str(p) for p in self.project_root.rglob(cache_dir) 
                if p.is_dir() and '.venv' not in str(p)
            )
        
        # Find empty directories
        for dir_path in self.project_root.rglob("*"):
            if (dir_path.is_dir() and 
                dir_path.name not in [".git", ".venv", ".vscode", "__pycache__"]):
                try:
                    if not any(dir_path.iterdir()):
                        analysis["empty_dirs"].append(str(dir_path))
                except:
                    pass
        
        # Find redundant reports
        report_patterns = ["*CLEANUP*", "*REPORT*", "*ANALYSIS*", "*SUMMARY*"]
        for pattern in report_patterns:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file() and file_path.suffix in ['.md', '.json']:
                    # Keep only essential files
                    essential = [
                        "MAINTENANCE_CONFIG.md",
                        "GITIGNORE_IMPROVEMENTS_FINAL.md", 
                        "README.md",
                        "ADVANCED_CLEANUP_REPORT.json"
                    ]
                    if file_path.name not in essential:
                        analysis["redundant_reports"].append(str(file_path))
        
        # Find old documentation
        docs_dir = self.project_root / "docs"
        if docs_dir.exists():
            old_patterns = ["*PHASE*", "*DETAILED*", "*CALIBRATION*"]
            for pattern in old_patterns:
                analysis["old_docs"].extend(
                    str(p) for p in docs_dir.glob(pattern)
                )
        
        return analysis
    
    def generate_recommendations(self, analysis: Dict) -> List[str]:
        """Gera recomendações baseadas na análise"""
        recommendations = []
        
        if analysis["duplicate_files"]:
            recommendations.append(
                f"Remover {len(analysis['duplicate_files'])} arquivos duplicados"
            )
        
        if analysis["large_files"]:
            large_count = len(analysis["large_files"])
            recommendations.append(
                f"Revisar {large_count} arquivos grandes (>1MB)"
            )
        
        if analysis["temp_files"]:
            recommendations.append(
                f"Limpar {len(analysis['temp_files'])} arquivos temporários"
            )
        
        if analysis["cache_dirs"]:
            recommendations.append(
                f"Remover {len(analysis['cache_dirs'])} diretórios de cache"
            )
        
        if analysis["empty_dirs"]:
            recommendations.append(
                f"Remover {len(analysis['empty_dirs'])} diretórios vazios"
            )
        
        if analysis["redundant_reports"]:
            recommendations.append(
                f"Consolidar {len(analysis['redundant_reports'])} relatórios redundantes"
            )
        
        if analysis["old_docs"]:
            recommendations.append(
                f"Arquivar {len(analysis['old_docs'])} documentos antigos"
            )
        
        return recommendations
    
    def execute_cleanup(self, analysis: Dict, dry_run: bool = False) -> Dict:
        """Executa limpeza baseada na análise"""
        cleanup_results = {
            "removed_files": [],
            "removed_dirs": [],
            "errors": [],
            "summary": {}
        }
        
        if dry_run:
            print("🔍 MODO SIMULAÇÃO - Nenhum arquivo será removido")
        
        # Clean temp files
        for temp_file in analysis["temp_files"]:
            try:
                if not dry_run:
                    Path(temp_file).unlink()
                cleanup_results["removed_files"].append(temp_file)
                print(f"   ✅ {Path(temp_file).name}")
            except Exception as e:
                cleanup_results["errors"].append(f"Erro ao remover {temp_file}: {e}")
        
        # Clean cache directories
        for cache_dir in analysis["cache_dirs"]:
            try:
                if not dry_run:
                    shutil.rmtree(cache_dir)
                cleanup_results["removed_dirs"].append(cache_dir)
                print(f"   ✅ {Path(cache_dir).name}/")
            except Exception as e:
                cleanup_results["errors"].append(f"Erro ao remover {cache_dir}: {e}")
        
        # Clean empty directories
        for empty_dir in analysis["empty_dirs"]:
            try:
                if not dry_run:
                    Path(empty_dir).rmdir()
                cleanup_results["removed_dirs"].append(empty_dir)
                print(f"   ✅ {Path(empty_dir).name}/")
            except Exception as e:
                cleanup_results["errors"].append(f"Erro ao remover {empty_dir}: {e}")
        
        # Clean redundant reports (with confirmation)
        if analysis["redundant_reports"]:
            print(f"\n⚠️  Encontrados {len(analysis['redundant_reports'])} relatórios redundantes")
            for report in analysis["redundant_reports"][:5]:  # Show first 5
                print(f"   - {Path(report).name}")
            
            if not dry_run:
                confirm = input("\nRemover relatórios redundantes? (y/N): ").lower()
                if confirm == 'y':
                    for report in analysis["redundant_reports"]:
                        try:
                            Path(report).unlink()
                            cleanup_results["removed_files"].append(report)
                            print(f"   ✅ {Path(report).name}")
                        except Exception as e:
                            cleanup_results["errors"].append(f"Erro ao remover {report}: {e}")
        
        # Generate summary
        cleanup_results["summary"] = {
            "total_files_removed": len(cleanup_results["removed_files"]),
            "total_dirs_removed": len(cleanup_results["removed_dirs"]),
            "total_errors": len(cleanup_results["errors"])
        }
        
        return cleanup_results
    
    def run_analysis_only(self) -> Dict:
        """Executa apenas análise sem limpeza"""
        print("🔍 ANÁLISE DO PROJETO - Photo Culling System")
        print("=" * 50)
        
        analysis = self.analyze_project_structure()
        recommendations = self.generate_recommendations(analysis)
        
        # Display results
        print(f"\n📊 RESULTADOS DA ANÁLISE:")
        print(f"   Arquivos duplicados: {len(analysis['duplicate_files'])}")
        print(f"   Arquivos grandes: {len(analysis['large_files'])}")
        print(f"   Arquivos temporários: {len(analysis['temp_files'])}")
        print(f"   Diretórios de cache: {len(analysis['cache_dirs'])}")
        print(f"   Diretórios vazios: {len(analysis['empty_dirs'])}")
        print(f"   Relatórios redundantes: {len(analysis['redundant_reports'])}")
        print(f"   Documentos antigos: {len(analysis['old_docs'])}")
        
        if recommendations:
            print(f"\n💡 RECOMENDAÇÕES:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Save analysis
        report_path = self.project_root / "PROJECT_ANALYSIS.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis,
                "recommendations": recommendations
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Análise salva em: {report_path}")
        return analysis
    
    def run_full_cleanup(self, dry_run: bool = False) -> Dict:
        """Executa análise completa + limpeza"""
        print("🧹 LIMPEZA COMPLETA DO PROJETO")
        print("=" * 50)
        
        # Analysis
        analysis = self.analyze_project_structure()
        recommendations = self.generate_recommendations(analysis)
        
        self.report["analysis"] = analysis
        self.report["recommendations"] = recommendations
        
        # Show analysis summary
        print(f"\n📊 ANÁLISE INICIAL:")
        for rec in recommendations:
            print(f"   • {rec}")
        
        # Execute cleanup
        if analysis["temp_files"] or analysis["cache_dirs"] or analysis["empty_dirs"]:
            print(f"\n🧹 EXECUTANDO LIMPEZA...")
            cleanup_results = self.execute_cleanup(analysis, dry_run)
            self.report["cleanup_results"] = cleanup_results
            
            print(f"\n📊 RESULTADOS:")
            print(f"   Arquivos removidos: {cleanup_results['summary']['total_files_removed']}")
            print(f"   Diretórios removidos: {cleanup_results['summary']['total_dirs_removed']}")
            if cleanup_results["errors"]:
                print(f"   Erros: {cleanup_results['summary']['total_errors']}")
        
        # Save final report
        report_path = self.project_root / "UNIFIED_CLEANUP_REPORT.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Relatório completo salvo em: {report_path}")
        return self.report

def main():
    """Função principal com interface de linha de comando"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Ferramenta Unificada de Limpeza - Photo Culling System'
    )
    parser.add_argument('--analyze', action='store_true', 
                       help='Apenas analisar sem fazer limpeza')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simular limpeza sem remover arquivos')
    parser.add_argument('--project-root', default='.',
                       help='Diretório raiz do projeto')
    
    args = parser.parse_args()
    
    cleanup_tool = UnifiedCleanupTool(args.project_root)
    
    if args.analyze:
        cleanup_tool.run_analysis_only()
    else:
        cleanup_tool.run_full_cleanup(dry_run=args.dry_run)

if __name__ == "__main__":
    main()
