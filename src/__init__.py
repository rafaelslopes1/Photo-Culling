"""
Photo Culling AI System - Consolidated Package
Sistema consolidado de classificação automática de imagens

Este pacote consolidado combina todas as funcionalidades:
- Extração de características de imagens
- Interface web para rotulagem manual  
- Sistema de IA para classificação automática
- Processamento em lote de imagens
- Utilitários de manutenção de dados
"""

__version__ = "2.0.0"
__author__ = "Photo Culling AI Team"

# Core functionality
from .core import (
    FeatureExtractor, 
    extract_features_from_folder,
    AIClassifier,
    train_classifier_from_folder, 
    ImageProcessor,
    process_images_with_ai,
    process_images_basic
)

# Web interface
from .web import (
    WebLabelingApp,
    create_app,
    start_web_server
)

# Utilities
from .utils import (
    DataUtils,
    clean_all_data,
    create_backup,
    get_system_statistics,
    ConfigManager, 
    get_config,
    get_setting
)

__all__ = [
    # Core
    'FeatureExtractor',
    'extract_features_from_folder',
    'AIClassifier', 
    'train_classifier_from_folder',
    'ImageProcessor',
    'process_images_with_ai',
    'process_images_basic',
    
    # Web
    'WebLabelingApp',    'create_app',
    'start_web_server',
    
    # Utils
    'DataUtils',
    'clean_all_data',
    'create_backup',
    'get_system_statistics',
    'ConfigManager',
    'get_config',
    'get_setting'
]
