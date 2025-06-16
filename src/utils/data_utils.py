#!/usr/bin/env python3
"""
Data Utilities for Photo Culling System
Utilitários consolidados para limpeza, backup e manipulação de dados
"""

import os
import shutil
import sqlite3
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories():
    """
    Setup required directories for the photo culling system.
    """
    directories = [
        'data',
        'data/input',
        'data/features',
        'data/labels',
        'data/models',
        'data/backups',
        'docs',
        'tools'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info("📁 Diretórios configurados com sucesso")

class DataUtils:
    """
    Utilitários consolidados para manipulação de dados do sistema
    Combina funções de limpeza, backup e manutenção
    """
    
    def __init__(self, labels_db="data/labels/labels.db", 
                 features_db="data/features/features.db",
                 backup_dir="data/backups"):
        
        self.labels_db = labels_db
        self.features_db = features_db
        self.backup_dir = backup_dir
        
        os.makedirs(backup_dir, exist_ok=True)
    
    def clean_labels(self, remove_duplicates=True, remove_orphans=True):
        """
        Limpa banco de dados de rótulos
        
        Args:
            remove_duplicates: Remove rótulos duplicados
            remove_orphans: Remove rótulos de imagens inexistentes
            
        Returns:
            dict: Estatísticas da limpeza
        """
        logger.info("🧹 Iniciando limpeza dos rótulos...")
        
        stats = {
            'duplicates_removed': 0,
            'orphans_removed': 0,
            'invalid_removed': 0,
            'total_before': 0,
            'total_after': 0
        }
        
        try:
            conn = sqlite3.connect(self.labels_db)
            cursor = conn.cursor()
            
            # Count initial records
            cursor.execute('SELECT COUNT(*) FROM labels')
            stats['total_before'] = cursor.fetchone()[0]
            
            if remove_duplicates:
                # Remove duplicate labels (keep most recent)
                cursor.execute('''
                    DELETE FROM labels 
                    WHERE id NOT IN (
                        SELECT MAX(id) 
                        FROM labels 
                        GROUP BY filename
                    )
                ''')
                stats['duplicates_removed'] = cursor.rowcount
                logger.info(f"✓ Removed {stats['duplicates_removed']} duplicate labels")
            
            if remove_orphans:
                # Get all labeled filenames
                cursor.execute('SELECT DISTINCT filename FROM labels')
                labeled_files = [row[0] for row in cursor.fetchall()]
                
                # Check which files don't exist
                orphans = []
                for filename in labeled_files:
                    # Check in various possible locations
                    possible_paths = [
                        f"data/input/{filename}",
                        f"input/{filename}",
                        filename
                    ]
                    
                    if not any(os.path.exists(path) for path in possible_paths):
                        orphans.append(filename)
                
                # Remove orphan labels
                if orphans:
                    placeholders = ','.join(['?' for _ in orphans])
                    cursor.execute(f'DELETE FROM labels WHERE filename IN ({placeholders})', orphans)
                    stats['orphans_removed'] = len(orphans)
                    logger.info(f"✓ Removed {stats['orphans_removed']} orphan labels")
            
            # Remove invalid labels (missing required fields)
            cursor.execute('''
                DELETE FROM labels 
                WHERE (label_type = 'quality' AND score IS NULL)
                   OR (label_type = 'rejection' AND rejection_reason IS NULL)
                   OR label_type NOT IN ('quality', 'rejection')
            ''')
            stats['invalid_removed'] = cursor.rowcount
            logger.info(f"✓ Removed {stats['invalid_removed']} invalid labels")
            
            # Count final records
            cursor.execute('SELECT COUNT(*) FROM labels')
            stats['total_after'] = cursor.fetchone()[0]
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Limpeza concluída: {stats['total_before']} → {stats['total_after']} rótulos")
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Erro na limpeza: {e}")
            return stats
    
    def backup_data(self, include_images=False):
        """
        Cria backup completo dos dados
        
        Args:
            include_images: Se incluir as imagens no backup
            
        Returns:
            str: Caminho do backup criado
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(self.backup_dir, f"backup_{timestamp}")
        
        logger.info(f"💾 Criando backup em {backup_path}...")
        
        os.makedirs(backup_path, exist_ok=True)
        
        try:
            # Backup databases
            if os.path.exists(self.labels_db):
                shutil.copy2(self.labels_db, os.path.join(backup_path, "labels.db"))
                logger.info("✓ Labels database backed up")
            
            if os.path.exists(self.features_db):
                shutil.copy2(self.features_db, os.path.join(backup_path, "features.db"))
                logger.info("✓ Features database backed up")
            
            # Backup config if exists
            if os.path.exists("config.json"):
                shutil.copy2("config.json", os.path.join(backup_path, "config.json"))
                logger.info("✓ Config backed up")
            
            # Export to JSON for portability
            self._export_labels_to_json(os.path.join(backup_path, "labels_export.json"))
            
            # Backup images if requested
            if include_images:
                images_dir = "data/input"
                if os.path.exists(images_dir):
                    backup_images_dir = os.path.join(backup_path, "images")
                    shutil.copytree(images_dir, backup_images_dir)
                    logger.info("✓ Images backed up")
            
            # Create backup info
            backup_info = {
                'timestamp': timestamp,
                'labels_db': os.path.exists(self.labels_db),
                'features_db': os.path.exists(self.features_db),
                'images_included': include_images,
                'created_by': 'DataUtils.backup_data'
            }
            
            with open(os.path.join(backup_path, "backup_info.json"), 'w') as f:
                json.dump(backup_info, f, indent=2)
            
            logger.info(f"✅ Backup criado com sucesso: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"❌ Erro criando backup: {e}")
            return None
    
    def restore_from_backup(self, backup_path):
        """
        Restaura dados de um backup
        
        Args:
            backup_path: Caminho do backup
            
        Returns:
            bool: Sucesso da operação
        """
        logger.info(f"📥 Restaurando backup de {backup_path}...")
        
        if not os.path.exists(backup_path):
            logger.error(f"Backup não encontrado: {backup_path}")
            return False
        
        try:
            # Restore databases
            labels_backup = os.path.join(backup_path, "labels.db")
            if os.path.exists(labels_backup):
                os.makedirs(os.path.dirname(self.labels_db), exist_ok=True)
                shutil.copy2(labels_backup, self.labels_db)
                logger.info("✓ Labels database restored")
            
            features_backup = os.path.join(backup_path, "features.db")
            if os.path.exists(features_backup):
                os.makedirs(os.path.dirname(self.features_db), exist_ok=True)
                shutil.copy2(features_backup, self.features_db)
                logger.info("✓ Features database restored")
            
            # Restore config
            config_backup = os.path.join(backup_path, "config.json")
            if os.path.exists(config_backup):
                shutil.copy2(config_backup, "config.json")
                logger.info("✓ Config restored")
            
            logger.info("✅ Backup restaurado com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro restaurando backup: {e}")
            return False
    
    def _export_labels_to_json(self, output_path):
        """Exporta rótulos para JSON"""
        try:
            conn = sqlite3.connect(self.labels_db)
            df = pd.read_sql_query('SELECT * FROM labels', conn)
            conn.close()
            
            df.to_json(output_path, orient='records', indent=2)
            logger.info(f"✓ Labels exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting labels: {e}")
    
    def get_data_statistics(self):
        """
        Obtém estatísticas dos dados
        
        Returns:
            dict: Estatísticas detalhadas
        """
        stats = {
            'labels': {},
            'features': {},
            'images': {},
            'data_integrity': {}
        }
        
        try:
            # Labels statistics
            if os.path.exists(self.labels_db):
                conn = sqlite3.connect(self.labels_db)
                cursor = conn.cursor()
                
                cursor.execute('SELECT COUNT(*) FROM labels')
                stats['labels']['total'] = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(DISTINCT filename) FROM labels')
                stats['labels']['unique_files'] = cursor.fetchone()[0]
                
                cursor.execute('SELECT label_type, COUNT(*) FROM labels GROUP BY label_type')
                stats['labels']['by_type'] = dict(cursor.fetchall())
                
                cursor.execute('SELECT score, COUNT(*) FROM labels WHERE label_type="quality" GROUP BY score')
                stats['labels']['quality_distribution'] = dict(cursor.fetchall())
                
                conn.close()
            
            # Features statistics
            if os.path.exists(self.features_db):
                conn = sqlite3.connect(self.features_db)
                cursor = conn.cursor()
                
                cursor.execute('SELECT COUNT(*) FROM image_features')
                stats['features']['total'] = cursor.fetchone()[0]
                
                cursor.execute('PRAGMA table_info(image_features)')
                columns = cursor.fetchall()
                stats['features']['feature_count'] = len([c for c in columns if c[1] != 'filename'])
                
                conn.close()
            
            # Images statistics
            images_dir = "data/input"
            if os.path.exists(images_dir):
                extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
                image_count = 0
                total_size = 0
                
                for ext in extensions:
                    for img_path in Path(images_dir).glob(f"*{ext}"):
                        image_count += 1
                        total_size += img_path.stat().st_size
                    for img_path in Path(images_dir).glob(f"*{ext.upper()}"):
                        image_count += 1
                        total_size += img_path.stat().st_size
                
                stats['images']['count'] = image_count
                stats['images']['total_size_mb'] = round(total_size / (1024 * 1024), 2)
            
            # Data integrity checks
            if stats['labels'].get('total', 0) > 0 and stats['features'].get('total', 0) > 0:
                labeled_count = stats['labels']['unique_files']
                features_count = stats['features']['total']
                
                stats['data_integrity']['labels_with_features'] = min(labeled_count, features_count)
                stats['data_integrity']['labels_without_features'] = max(0, labeled_count - features_count)
                stats['data_integrity']['features_without_labels'] = max(0, features_count - labeled_count)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return stats
    
    def optimize_databases(self):
        """Otimiza bancos de dados (VACUUM, REINDEX)"""
        logger.info("⚡ Otimizando bancos de dados...")
        
        optimized = []
        
        for db_path in [self.labels_db, self.features_db]:
            if os.path.exists(db_path):
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute('VACUUM')
                    cursor.execute('REINDEX')
                    
                    conn.close()
                    optimized.append(os.path.basename(db_path))
                    
                except Exception as e:
                    logger.error(f"Error optimizing {db_path}: {e}")
        
        logger.info(f"✅ Databases optimized: {optimized}")
        return optimized

# Convenience functions
def clean_all_data(labels_db="data/labels/labels.db", 
                  features_db="data/features/features.db"):
    """
    Função de conveniência para limpeza completa dos dados
    
    Args:
        labels_db: Caminho do banco de rótulos
        features_db: Caminho do banco de features
        
    Returns:
        dict: Estatísticas da limpeza
    """
    utils = DataUtils(labels_db, features_db)
    
    # Create backup first
    backup_path = utils.backup_data()
    if backup_path:
        logger.info(f"✓ Backup criado antes da limpeza: {backup_path}")
    
    # Clean labels
    clean_stats = utils.clean_labels()
    
    # Optimize databases
    utils.optimize_databases()
    
    return clean_stats

def create_backup(include_images=False):
    """
    Função de conveniência para criar backup
    
    Args:
        include_images: Se incluir imagens no backup
        
    Returns:
        str: Caminho do backup criado
    """
    utils = DataUtils()
    return utils.backup_data(include_images)

def get_system_statistics():
    """
    Função de conveniência para obter estatísticas do sistema
    
    Returns:
        dict: Estatísticas completas
    """
    utils = DataUtils()
    return utils.get_data_statistics()

if __name__ == "__main__":
    # Example usage
    print("🛠️ Data Utils - Sistema de Utilitários")
    print("=" * 40)
    
    utils = DataUtils()
    
    # Get statistics
    stats = utils.get_data_statistics()
    print(f"📊 Estatísticas:")
    print(f"  Rótulos: {stats['labels'].get('total', 0)}")
    print(f"  Features: {stats['features'].get('total', 0)}")
    print(f"  Imagens: {stats['images'].get('count', 0)}")
    
    # Offer options
    print("\nOpções disponíveis:")
    print("1. Limpar dados")
    print("2. Criar backup")
    print("3. Otimizar bancos")
    print("4. Mostrar estatísticas detalhadas")
    
    choice = input("\nEscolha uma opção (1-4): ").strip()
    
    if choice == '1':
        clean_stats = utils.clean_labels()
        print(f"✅ Limpeza concluída: {clean_stats}")
    elif choice == '2':
        backup_path = utils.backup_data()
        print(f"✅ Backup criado: {backup_path}")
    elif choice == '3':
        optimized = utils.optimize_databases()
        print(f"✅ Bancos otimizados: {optimized}")
    elif choice == '4':
        print("\n📊 Estatísticas Detalhadas:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
