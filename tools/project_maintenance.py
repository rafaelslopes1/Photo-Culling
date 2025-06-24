#!/usr/bin/env python3
"""
Project Maintenance Tool
Automated maintenance script for Photo Culling System
Performs regular cleanup and monitoring tasks
"""

import os
import sys
import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProjectMaintenance:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.maintenance_log = self.project_root / "maintenance.log"
        
    def check_large_files(self, size_limit_mb: int = 10) -> List[Dict]:
        """Check for large files that might need attention"""
        large_files = []
        size_limit_bytes = size_limit_mb * 1024 * 1024
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip git and cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                file_path = Path(root) / file
                try:
                    file_size = file_path.stat().st_size
                    if file_size > size_limit_bytes:
                        large_files.append({
                            'path': str(file_path.relative_to(self.project_root)),
                            'size_mb': round(file_size / (1024 * 1024), 2)
                        })
                except (OSError, FileNotFoundError):
                    continue
                    
        return large_files
    
    def check_database_sizes(self) -> Dict[str, float]:
        """Check database file sizes"""
        db_sizes = {}
        db_paths = [
            self.project_root / "data" / "features" / "features.db",
            self.project_root / "data" / "labels" / "labels.db",
            self.project_root / "data" / "quality" / "quality.db"
        ]
        
        for db_path in db_paths:
            if db_path.exists():
                size_mb = db_path.stat().st_size / (1024 * 1024)
                db_sizes[str(db_path.relative_to(self.project_root))] = round(size_mb, 2)
                
        return db_sizes
    
    def check_temp_files(self) -> List[str]:
        """Find temporary files that can be cleaned"""
        temp_patterns = ['.tmp', '.temp', '.log', '.bak', '.backup']
        temp_files = []
        
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if any(file.endswith(pattern) for pattern in temp_patterns):
                    file_path = Path(root) / file
                    temp_files.append(str(file_path.relative_to(self.project_root)))
                    
        return temp_files
    
    def check_pycache_directories(self) -> List[str]:
        """Find __pycache__ directories"""
        pycache_dirs = []
        
        for root, dirs, files in os.walk(self.project_root):
            if '__pycache__' in dirs:
                pycache_path = Path(root) / '__pycache__'
                pycache_dirs.append(str(pycache_path.relative_to(self.project_root)))
                
        return pycache_dirs
    
    def check_git_status(self) -> Dict[str, List[str]]:
        """Check git repository status"""
        git_status = {
            'untracked_files': [],
            'modified_files': [],
            'staged_files': []
        }
        
        try:
            import subprocess
            
            # Check for untracked files
            result = subprocess.run(['git', 'ls-files', '--others', '--exclude-standard'], 
                                  capture_output=True, text=True, cwd=self.project_root)
            if result.returncode == 0:
                git_status['untracked_files'] = result.stdout.strip().split('\n') if result.stdout.strip() else []
                
            # Check for modified files
            result = subprocess.run(['git', 'diff', '--name-only'], 
                                  capture_output=True, text=True, cwd=self.project_root)
            if result.returncode == 0:
                git_status['modified_files'] = result.stdout.strip().split('\n') if result.stdout.strip() else []
                
            # Check for staged files
            result = subprocess.run(['git', 'diff', '--cached', '--name-only'], 
                                  capture_output=True, text=True, cwd=self.project_root)
            if result.returncode == 0:
                git_status['staged_files'] = result.stdout.strip().split('\n') if result.stdout.strip() else []
                
        except Exception as e:
            logger.warning(f"Erro ao verificar status do git: {e}")
            
        return git_status
    
    def clean_pycache(self) -> int:
        """Clean all __pycache__ directories"""
        import shutil
        cleaned_count = 0
        
        for root, dirs, files in os.walk(self.project_root):
            if '__pycache__' in dirs:
                pycache_path = Path(root) / '__pycache__'
                try:
                    shutil.rmtree(pycache_path)
                    cleaned_count += 1
                    logger.info(f"Removido: {pycache_path.relative_to(self.project_root)}")
                except Exception as e:
                    logger.error(f"Erro ao remover {pycache_path}: {e}")
                    
        return cleaned_count
    
    def clean_temp_files(self) -> int:
        """Clean temporary files"""
        temp_files = self.check_temp_files()
        cleaned_count = 0
        
        for temp_file in temp_files:
            temp_path = self.project_root / temp_file
            try:
                temp_path.unlink()
                cleaned_count += 1
                logger.info(f"Removido arquivo temporário: {temp_file}")
            except Exception as e:
                logger.error(f"Erro ao remover {temp_file}: {e}")
                
        return cleaned_count
    
    def generate_maintenance_report(self) -> Dict:
        """Generate comprehensive maintenance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'large_files': self.check_large_files(),
            'database_sizes': self.check_database_sizes(),
            'temp_files': self.check_temp_files(),
            'pycache_dirs': self.check_pycache_directories(),
            'git_status': self.check_git_status()
        }
        
        # Calculate totals first
        total_db_size_mb = sum(report['database_sizes'].values())
        
        # Calculate totals and recommendations
        report['summary'] = {
            'total_large_files': len(report['large_files']),
            'total_temp_files': len(report['temp_files']),
            'total_pycache_dirs': len(report['pycache_dirs']),
            'total_db_size_mb': total_db_size_mb,
            'recommendations': self._generate_recommendations(report, total_db_size_mb)
        }
        
        return report
    
    def _generate_recommendations(self, report: Dict, total_db_size: float) -> List[str]:
        """Generate maintenance recommendations"""
        recommendations = []
        
        if len(report['large_files']) > 0:
            recommendations.append("Considere mover arquivos grandes para .gitignore ou armazenamento externo")
            
        if len(report['temp_files']) > 10:
            recommendations.append("Execute limpeza de arquivos temporários")
            
        if len(report['pycache_dirs']) > 5:
            recommendations.append("Execute limpeza de diretórios __pycache__")
            
        if total_db_size > 100:
            recommendations.append("Considere otimizar ou arquivar bancos de dados grandes")
            
        if len(report['git_status']['untracked_files']) > 20:
            recommendations.append("Verifique arquivos não rastreados pelo git")
            
        return recommendations
    
    def run_maintenance(self, clean: bool = False) -> Dict:
        """Run maintenance tasks"""
        logger.info("Iniciando manutenção do projeto...")
        
        # Generate report
        report = self.generate_maintenance_report()
        
        # Perform cleaning if requested
        if clean:
            logger.info("Executando limpeza...")
            cleaned_pycache = self.clean_pycache()
            cleaned_temp = self.clean_temp_files()
            
            report['cleaning_results'] = {
                'pycache_cleaned': cleaned_pycache,
                'temp_files_cleaned': cleaned_temp
            }
        
        # Save report
        report_path = self.project_root / "MAINTENANCE_REPORT.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Relatório de manutenção salvo em: {report_path}")
        
        return report

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Photo Culling System Maintenance Tool')
    parser.add_argument('--clean', action='store_true', help='Perform cleaning operations')
    parser.add_argument('--project-root', default='.', help='Project root directory')
    
    args = parser.parse_args()
    
    maintenance = ProjectMaintenance(args.project_root)
    report = maintenance.run_maintenance(clean=args.clean)
    
    # Print summary
    print(f"\n{'='*50}")
    print("RELATÓRIO DE MANUTENÇÃO")
    print(f"{'='*50}")
    print(f"Timestamp: {report['timestamp']}")
    print(f"Arquivos grandes: {report['summary']['total_large_files']}")
    print(f"Arquivos temporários: {report['summary']['total_temp_files']}")
    print(f"Diretórios __pycache__: {report['summary']['total_pycache_dirs']}")
    print(f"Tamanho total dos bancos (MB): {report['summary']['total_db_size_mb']}")
    
    if report['summary']['recommendations']:
        print(f"\nRECOMENDAÇÕES:")
        for i, rec in enumerate(report['summary']['recommendations'], 1):
            print(f"{i}. {rec}")
    
    if args.clean and 'cleaning_results' in report:
        print(f"\nLIMPEZA REALIZADA:")
        print(f"Diretórios __pycache__ removidos: {report['cleaning_results']['pycache_cleaned']}")
        print(f"Arquivos temporários removidos: {report['cleaning_results']['temp_files_cleaned']}")

if __name__ == "__main__":
    main()
