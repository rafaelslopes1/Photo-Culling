#!/usr/bin/env python3
"""
Configuration Manager for Photo Culling System
Gerenciador de configurações consolidado para o sistema
"""

import json
import os
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Gerenciador centralizado de configurações do sistema
    Permite carregamento, validação e salvamento de configurações
    """
    
    DEFAULT_CONFIG = {
        "system": {
            "version": "2.0_consolidated",
            "debug": False,
            "log_level": "INFO"
        },
        "paths": {
            "input_folder": "data/input",
            "labels_db": "data/labels/labels.db",
            "features_db": "data/features/features.db", 
            "models_dir": "data/models",
            "backup_dir": "data/backups"
        },
        "processing": {
            "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
            "multiprocessing": {
                "enabled": True,
                "max_workers": None,
                "chunk_size": 4
            },
            "quality_thresholds": {
                "blur_threshold": 25,
                "brightness_threshold": 40,
                "contrast_threshold": 20
            },
            "quality_weights": {
                "sharpness": 1.0,
                "brightness": 1.0,
                "contrast": 0.5,
                "color_harmony": 0.3
            }
        },
        "ai": {
            "enabled": True,
            "auto_retrain": False,
            "min_samples_per_class": 10,
            "confidence_threshold": 0.7,
            "models": {
                "default_algorithm": "random_forest",
                "cross_validation_folds": 3,
                "test_size": 0.2
            }
        },
        "web_interface": {
            "host": "localhost",
            "port": 5002,
            "debug": True,
            "auto_ai_suggestions": True,
            "keyboard_shortcuts": {
                "quality_1": "1",
                "quality_2": "2", 
                "quality_3": "3",
                "quality_4": "4",
                "quality_5": "5",
                "reject_blur": "b",
                "reject_dark": "d",
                "reject_light": "l",
                "reject_cropped": "c",
                "reject_other": "x",
                "next_image": "space",
                "prev_image": "backspace",
                "show_info": "i"
            }
        },
        "output": {
            "folders": {
                "selected": "selected",
                "duplicates": "duplicates",
                "blurry": "blurry", 
                "low_light": "low_light",
                "failed": "failed"
            },
            "naming": {
                "add_quality_prefix": True,
                "add_timestamp": False,
                "preserve_original": True
            }
        },
        "features": {
            "basic_enabled": True,
            "advanced_enabled": True,
            "face_detection": {
                "enabled": True,
                "min_face_size": 30,
                "confidence_threshold": 0.8
            },
            "color_analysis": {
                "dominant_colors_count": 5,
                "color_temperature": True,
                "color_harmony": True
            },
            "composition": {
                "rule_of_thirds": True,
                "symmetry": True,
                "edge_density": True
            }
        }
    }
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()
        self.load_config()
    
    def load_config(self) -> bool:
        """
        Carrega configuração do arquivo
        
        Returns:
            bool: Sucesso da operação
        """
        if not os.path.exists(self.config_path):
            logger.info(f"Config file not found, creating default: {self.config_path}")
            return self.save_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            
            # Merge with defaults (preserves new default keys)
            self.config = self._deep_merge(self.DEFAULT_CONFIG, loaded_config)
            
            # Validate config
            if self._validate_config():
                logger.info(f"✓ Configuration loaded from {self.config_path}")
                return True
            else:
                logger.warning("⚠️ Configuration validation failed, using defaults")
                self.config = self.DEFAULT_CONFIG.copy()
                return False
                
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            logger.info("Using default configuration")
            self.config = self.DEFAULT_CONFIG.copy()
            return False
    
    def save_config(self) -> bool:
        """
        Salva configuração no arquivo
        
        Returns:
            bool: Sucesso da operação
        """
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(self.config_path) or '.', exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            
            logger.info(f"✓ Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving config: {e}")
            return False
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Obtém valor de configuração usando notação de ponto
        
        Args:
            key_path: Caminho da chave (ex: "ai.models.default_algorithm")
            default: Valor padrão se não encontrado
            
        Returns:
            Any: Valor da configuração
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Define valor de configuração usando notação de ponto
        
        Args:
            key_path: Caminho da chave (ex: "ai.enabled")
            value: Novo valor
        """
        keys = key_path.split('.')
        config_ref = self.config
        
        # Navigate to parent
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        # Set value
        config_ref[keys[-1]] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Obtém seção completa da configuração
        
        Args:
            section: Nome da seção
            
        Returns:
            dict: Configurações da seção
        """
        return self.config.get(section, {})
    
    def update_section(self, section: str, updates: Dict[str, Any]) -> None:
        """
        Atualiza seção da configuração
        
        Args:
            section: Nome da seção
            updates: Atualizações a aplicar
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section].update(updates)
    
    def _deep_merge(self, base: Dict, updates: Dict) -> Dict:
        """Merge profundo de dicionários"""
        result = base.copy()
        
        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self) -> bool:
        """
        Valida configuração carregada
        
        Returns:
            bool: Se configuração é válida
        """
        try:
            # Check required sections
            required_sections = ['paths', 'processing', 'ai', 'web_interface']
            for section in required_sections:
                if section not in self.config:
                    logger.error(f"Missing required config section: {section}")
                    return False
            
            # Validate paths
            paths = self.config.get('paths', {})
            for path_key, path_value in paths.items():
                if path_key.endswith('_dir'):
                    # Create directory if it doesn't exist
                    os.makedirs(path_value, exist_ok=True)
            
            # Validate numeric values
            processing = self.config.get('processing', {})
            thresholds = processing.get('quality_thresholds', {})
            
            for threshold_key, threshold_value in thresholds.items():
                if not isinstance(threshold_value, (int, float)) or threshold_value < 0:
                    logger.error(f"Invalid threshold value: {threshold_key} = {threshold_value}")
                    return False
            
            # Validate web interface
            web = self.config.get('web_interface', {})
            port = web.get('port')
            if not isinstance(port, int) or port < 1024 or port > 65535:
                logger.error(f"Invalid port: {port}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Config validation error: {e}")
            return False
    
    def reset_to_defaults(self) -> None:
        """Reseta configuração para valores padrão"""
        self.config = self.DEFAULT_CONFIG.copy()
        logger.info("✓ Configuration reset to defaults")
    
    def create_backup(self) -> Optional[str]:
        """
        Cria backup da configuração atual
        
        Returns:
            str: Caminho do backup ou None se erro
        """
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"{self.config_path}.backup_{timestamp}"
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            
            logger.info(f"✓ Config backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"❌ Error creating config backup: {e}")
            return None
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtém resumo da configuração atual
        
        Returns:
            dict: Resumo das configurações principais
        """
        return {
            'system_version': self.get('system.version'),
            'ai_enabled': self.get('ai.enabled'),
            'web_port': self.get('web_interface.port'),
            'input_folder': self.get('paths.input_folder'),
            'multiprocessing': self.get('processing.multiprocessing.enabled'),
            'feature_extraction': {
                'basic': self.get('features.basic_enabled'),
                'advanced': self.get('features.advanced_enabled'),
                'face_detection': self.get('features.face_detection.enabled')
            },
            'quality_thresholds': self.get('processing.quality_thresholds'),
            'output_folders': list(self.get('output.folders', {}).values())
        }

# Global config instance
_config_instance = None

def get_config(config_path: str = "config.json") -> ConfigManager:
    """
    Obtém instância global do gerenciador de configuração
    
    Args:
        config_path: Caminho do arquivo de configuração
        
    Returns:
        ConfigManager: Instância do gerenciador
    """
    global _config_instance
    
    if _config_instance is None or _config_instance.config_path != config_path:
        _config_instance = ConfigManager(config_path)
    
    return _config_instance

# Convenience functions
def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Função de conveniência para carregar configuração
    
    Args:
        config_path: Caminho do arquivo
        
    Returns:
        dict: Configuração carregada
    """
    config_manager = get_config(config_path)
    return config_manager.config

def save_config(config: Dict[str, Any], config_path: str = "config.json") -> bool:
    """
    Função de conveniência para salvar configuração
    
    Args:
        config: Configuração a salvar
        config_path: Caminho do arquivo
        
    Returns:
        bool: Sucesso da operação
    """
    config_manager = get_config(config_path)
    config_manager.config = config
    return config_manager.save_config()

def get_setting(key_path: str, default: Any = None) -> Any:
    """
    Função de conveniência para obter configuração
    
    Args:
        key_path: Caminho da chave
        default: Valor padrão
        
    Returns:
        Any: Valor da configuração
    """
    config_manager = get_config()
    return config_manager.get(key_path, default)

if __name__ == "__main__":
    # Example usage
    print("⚙️ Configuration Manager - Sistema de Configuração")
    print("=" * 50)
    
    # Load config
    config = get_config()
    
    # Show summary
    summary = config.get_summary()
    print("📋 Resumo da Configuração:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Interactive options
    print("\nOpções:")
    print("1. Mostrar configuração completa")
    print("2. Criar backup da configuração")
    print("3. Resetar para padrões")
    print("4. Salvar configuração atual")
    
    choice = input("\nEscolha uma opção (1-4): ").strip()
    
    if choice == '1':
        print("\n📄 Configuração Completa:")
        print(json.dumps(config.config, indent=2, ensure_ascii=False))
    elif choice == '2':
        backup_path = config.create_backup()
        print(f"✅ Backup criado: {backup_path}")
    elif choice == '3':
        config.reset_to_defaults()
        config.save_config()
        print("✅ Configuração resetada para padrões")
    elif choice == '4':
        success = config.save_config()
        print(f"✅ Configuração salva: {success}")
