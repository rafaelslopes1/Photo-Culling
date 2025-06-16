#!/usr/bin/env python3
"""
Photo Culling System - Main Application
Sistema principal de classificação e curadoria de imagens.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.image_processor import ImageProcessor
from src.utils.config_manager import ConfigManager
from src.utils.data_utils import setup_directories


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description='Photo Culling System')
    parser.add_argument('--extract-features', action='store_true',
                       help='Extract features from images')
    parser.add_argument('--train-model', action='store_true',
                       help='Train AI model with labeled data')
    parser.add_argument('--classify', action='store_true',
                       help='Classify images using trained model')
    parser.add_argument('--web-interface', action='store_true',
                       help='Launch web labeling interface')
    parser.add_argument('--input-dir', type=str, default='data/input',
                       help='Input directory for images')
    parser.add_argument('--port', type=int, default=5001,
                       help='Port for web interface (default: 5001)')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Configuration file path')
    parser.add_argument('--selection-mode', type=str, 
                       choices=['sequential', 'smart'], default='sequential',
                       help='Image selection strategy for labeling (default: sequential)')
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config_manager.load_config()
    config = config_manager.config
    
    # Initialize image processor
    processor = ImageProcessor(args.config)
    
    try:
        if args.extract_features:
            print("🔍 Extraindo características das imagens...")
            processor.extract_features(args.input_dir)
            print("✅ Extração de características concluída!")
            
        elif args.train_model:
            print("🤖 Treinando modelo de IA...")
            accuracy = processor.train_model()
            print(f"✅ Modelo treinado com precisão: {accuracy:.2%}")
            
        elif args.classify:
            print("📊 Classificando imagens...")
            results = processor.classify_images(args.input_dir)
            print(f"✅ Classificação concluída! {len(results)} imagens processadas.")
            
        elif args.web_interface:
            print("🌐 Iniciando interface web...")
            from src.web.app import create_app
            app = create_app(config, selection_mode=args.selection_mode)
            print(f"🔌 Acesse: http://localhost:{args.port}")
            print(f"🎯 Modo de seleção: {args.selection_mode}")
            app.run(host='0.0.0.0', port=args.port, debug=True)
            
        else:
            print("📋 Photo Culling System")
            print("Use --help para ver as opções disponíveis")
            print("\nOpções principais:")
            print("  --extract-features  : Extrair características das imagens")
            print("  --train-model      : Treinar modelo de IA")
            print("  --classify         : Classificar imagens")
            print("  --web-interface    : Interface web para rotulação")
            
    except Exception as e:
        print(f"❌ Erro: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
