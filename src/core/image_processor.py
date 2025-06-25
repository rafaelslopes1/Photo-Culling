#!/usr/bin/env python3
"""
Consolidated Image Processor for Photo Culling System
Processador de imagens consolidado para sistema de classificação
Combina pipeline de processamento, culling e organização automática
"""

import os
import cv2
import shutil
import json
from datetime import datetime
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
import numpy as np
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import optimized blur detection
from .image_quality_analyzer import ImageQualityAnalyzer
from .person_blur_analyzer import PersonBlurAnalyzer
from data.quality.blur_config import get_threshold_by_strategy, DEFAULT_PRACTICAL_THRESHOLD

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Sistema consolidado de processamento de imagens
    Combina detecção de qualidade, organização automática e pipeline de culling
    """
    
    def __init__(self, config_path="config.json"):
        self.config = self._load_config(config_path)
        self.face_cascade = self._load_face_detector()
        self.processed_count = 0
        self.results = {
            'total': 0,
            'selected': 0,
            'duplicates': 0,
            'blurry': 0,
            'low_light': 0,
            'failures': 0
        }
        
        # Initialize optimized quality analyzer
        self.quality_analyzer = ImageQualityAnalyzer()
        
        # Use optimized blur detection if configured
        self.use_optimized_blur = self.config.get('processing_settings', {}).get('blur_detection_optimized', {}).get('enabled', False)
        if self.use_optimized_blur:
            blur_strategy = self.config.get('processing_settings', {}).get('blur_detection_optimized', {}).get('strategy', 'balanced')
            self.blur_threshold = self._get_blur_threshold_by_strategy(blur_strategy)
        else:
            self.blur_threshold = self.config.get('processing_settings', {}).get('blur_threshold', DEFAULT_PRACTICAL_THRESHOLD)
        
    def extract_features(self, input_dir):
        """Extract features from images in input directory"""
        from .feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor()
        return extractor.extract_from_directory(input_dir)
    
    def train_model(self):
        """Train AI model with labeled data"""
        from .ai_classifier import AIClassifier
        
        classifier = AIClassifier()
        return classifier.train_model()
    
    def classify_images(self, input_dir):
        """Classify images using trained model"""
        from .ai_classifier import AIClassifier
        
        classifier = AIClassifier()
        return classifier.predict_directory(input_dir)
        
    def _load_config(self, config_path):
        """Carrega configurações do arquivo JSON"""
        default_config = {
            "processing_settings": {
                "blur_threshold": 25,
                "brightness_threshold": 40,
                "quality_score_weights": {
                    "sharpness": 1.0,
                    "brightness": 1.0
                },
                "image_extensions": [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"],
                "multiprocessing": {
                    "enabled": True,
                    "max_workers": None,
                    "chunk_size": 4
                }
            },
            "output_folders": {
                "selected": "selected",
                "duplicates": "duplicates",
                "blurry": "blurry",
                "low_light": "low_light",
                "failed": "failed"
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                return {**default_config, **config}
            except Exception as e:
                logger.warning(f"Erro carregando config: {e}. Usando padrões.")
        
        return default_config
    
    def _load_face_detector(self):
        """Carrega detector de faces"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            return cv2.CascadeClassifier(cascade_path)
        except:
            logger.warning("Detector de faces não disponível")
            return None
    
    def process_folder(self, input_folder, output_folder, use_ai=False, ai_classifier=None):
        """
        Processa uma pasta completa de imagens
        
        Args:
            input_folder: Pasta de entrada
            output_folder: Pasta de saída
            use_ai: Se usar classificação por IA
            ai_classifier: Instância do classificador (se use_ai=True)
            
        Returns:
            dict: Estatísticas do processamento
        """
        logger.info(f"🚀 Iniciando processamento de {input_folder}")
        
        # Create output folders
        self._create_output_folders(output_folder)
        
        # Find images
        image_paths = self._find_images(input_folder)
        
        if not image_paths:
            logger.warning(f"Nenhuma imagem encontrada em {input_folder}")
            return self.results
        
        logger.info(f"Encontradas {len(image_paths)} imagens para processar")
        self.results['total'] = len(image_paths)
        
        # Process images
        if self.config['processing_settings']['multiprocessing']['enabled']:
            self._process_images_parallel(image_paths, output_folder, use_ai, ai_classifier)
        else:
            self._process_images_sequential(image_paths, output_folder, use_ai, ai_classifier)
        
        # Generate report
        self._generate_processing_report(output_folder)
        
        logger.info("✅ Processamento concluído!")
        return self.results
    
    def _find_images(self, folder_path):
        """Encontra todas as imagens na pasta"""
        extensions = self.config['processing_settings']['image_extensions']
        image_paths = []
        
        for ext in extensions:
            pattern = f"*{ext}"
            image_paths.extend(Path(folder_path).glob(pattern))
            image_paths.extend(Path(folder_path).glob(pattern.upper()))
        
        return [str(p) for p in image_paths]
    
    def _create_output_folders(self, output_folder):
        """Cria pastas de saída"""
        folders = self.config['output_folders']
        
        for folder_name in folders.values():
            folder_path = os.path.join(output_folder, folder_name)
            os.makedirs(folder_path, exist_ok=True)
    
    def _process_images_parallel(self, image_paths, output_folder, use_ai, ai_classifier):
        """Processa imagens em paralelo"""
        max_workers = self.config['processing_settings']['multiprocessing']['max_workers']
        if max_workers is None:
            max_workers = min(4, len(image_paths))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._process_single_image, 
                    path, output_folder, use_ai, ai_classifier
                ): path 
                for path in image_paths
            }
            
            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    self._update_results(result)
                    self.processed_count += 1
                    
                    if self.processed_count % 10 == 0:
                        logger.info(f"Processadas {self.processed_count}/{len(image_paths)} imagens")
                        
                except Exception as e:
                    logger.error(f"Erro processando {path}: {e}")
                    self.results['failures'] += 1
    
    def _process_images_sequential(self, image_paths, output_folder, use_ai, ai_classifier):
        """Processa imagens sequencialmente"""
        for i, path in enumerate(image_paths, 1):
            try:
                result = self._process_single_image(path, output_folder, use_ai, ai_classifier)
                self._update_results(result)
                
                if i % 10 == 0:
                    logger.info(f"Processadas {i}/{len(image_paths)} imagens")
                    
            except Exception as e:
                logger.error(f"Erro processando {path}: {e}")
                self.results['failures'] += 1
    
    def _process_single_image(self, image_path, output_folder, use_ai, ai_classifier):
        """
        Processa uma única imagem
        
        Returns:
            str: Categoria da imagem ('selected', 'blurry', etc.)
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return 'failed'
            
            filename = os.path.basename(image_path)
            
            # Use AI classification if available
            if use_ai and ai_classifier:
                return self._classify_with_ai(image_path, filename, output_folder, ai_classifier)
            else:
                return self._classify_with_rules(image, filename, image_path, output_folder)
                
        except Exception as e:
            logger.error(f"Erro processando {image_path}: {e}")
            return 'failed'
    
    def _classify_with_ai(self, image_path, filename, output_folder, ai_classifier):
        """Classifica imagem usando IA"""
        try:
            prediction_result = ai_classifier.get_prediction_for_image(image_path)
            
            if not prediction_result:
                return self._fallback_to_rules(image_path, filename, output_folder)
            
            prediction = prediction_result['prediction']
            confidence = prediction_result['confidence']
            
            # High confidence quality predictions
            if prediction.startswith('quality_') and confidence > 0.7:
                quality_score = int(prediction.split('_')[1])
                
                if quality_score >= 4:  # High quality
                    destination = os.path.join(output_folder, self.config['output_folders']['selected'])
                    shutil.copy2(image_path, os.path.join(destination, filename))
                    return 'selected'
                else:  # Lower quality, but still acceptable
                    destination = os.path.join(output_folder, self.config['output_folders']['selected'])
                    shutil.copy2(image_path, os.path.join(destination, f"q{quality_score}_{filename}"))
                    return 'selected'
            
            # Rejection predictions
            elif prediction.startswith('reject_'):
                reason = prediction.split('_')[1]
                
                if reason == 'blur':
                    destination = os.path.join(output_folder, self.config['output_folders']['blurry'])
                    shutil.copy2(image_path, os.path.join(destination, filename))
                    return 'blurry'
                elif reason in ['dark', 'light']:
                    destination = os.path.join(output_folder, self.config['output_folders']['low_light'])
                    shutil.copy2(image_path, os.path.join(destination, filename))
                    return 'low_light'
                else:
                    # Other rejections go to failed folder
                    destination = os.path.join(output_folder, self.config['output_folders']['failed'])
                    shutil.copy2(image_path, os.path.join(destination, filename))
                    return 'failed'
            
            # Low confidence - fallback to rules
            else:
                return self._fallback_to_rules(image_path, filename, output_folder)
                
        except Exception as e:
            logger.error(f"Erro na classificação por IA: {e}")
            return self._fallback_to_rules(image_path, filename, output_folder)
    
    def _fallback_to_rules(self, image_path, filename, output_folder):
        """Fallback para classificação baseada em regras"""
        image = cv2.imread(image_path)
        return self._classify_with_rules(image, filename, image_path, output_folder)
    
    def _classify_with_rules(self, image, filename, image_path, output_folder):
        """Classifica imagem usando regras baseadas em métricas"""
        # Calculate quality metrics
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sharpness (blur detection) - use optimized threshold
        if self.use_optimized_blur:
            # Use optimized blur detection system
            analysis_result = self.quality_analyzer.analyze_single_image(image_path)
            blur_score = analysis_result['blur_score']
            is_blurry = blur_score < self.blur_threshold
            
            # Log blur detection details if debug enabled
            if self.config.get('processing_settings', {}).get('blur_detection_optimized', {}).get('debug', False):
                strategy = self.config.get('processing_settings', {}).get('blur_detection_optimized', {}).get('strategy', 'balanced')
                logger.debug(f"Blur detection - File: {filename}, Score: {blur_score:.2f}, Threshold: {self.blur_threshold} ({strategy}), Blurry: {is_blurry}")
        else:
            # Legacy blur detection
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            is_blurry = laplacian_var < self.config['processing_settings']['blur_threshold']
        
        # Brightness
        brightness = gray.mean()
        is_too_dark = brightness < self.config['processing_settings']['brightness_threshold']
        is_too_bright = brightness > (255 - self.config['processing_settings']['brightness_threshold'])
        
        # Classify and move
        if is_blurry:
            destination = os.path.join(output_folder, self.config['output_folders']['blurry'])
            shutil.copy2(image_path, os.path.join(destination, filename))
            return 'blurry'
        
        elif is_too_dark or is_too_bright:
            destination = os.path.join(output_folder, self.config['output_folders']['low_light'])
            shutil.copy2(image_path, os.path.join(destination, filename))
            return 'low_light'
        
        else:
            # Good quality - calculate score
            quality_score = self._calculate_quality_score(image)
            
            destination = os.path.join(output_folder, self.config['output_folders']['selected'])
            
            # Add quality prefix to filename
            quality_filename = f"q{quality_score:.1f}_{filename}"
            shutil.copy2(image_path, os.path.join(destination, quality_filename))
            
            return 'selected'
    
    def _calculate_quality_score(self, image):
        """Calcula score de qualidade baseado em métricas"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sharpness score (0-10)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(10, laplacian_var / 10)
        
        # Brightness score (0-10)
        brightness = gray.mean()
        brightness_score = 10 - abs(brightness - 127.5) / 12.75  # Optimal at 127.5
        brightness_score = max(0, brightness_score)
        
        # Contrast score (0-10)
        contrast = gray.std()
        contrast_score = min(10, contrast / 6.4)
        
        # Weighted combination
        weights = self.config['processing_settings']['quality_score_weights']
        total_score = (
            sharpness_score * weights.get('sharpness', 1.0) +
            brightness_score * weights.get('brightness', 1.0) +
            contrast_score * weights.get('contrast', 0.5)
        ) / sum([weights.get('sharpness', 1.0), weights.get('brightness', 1.0), weights.get('contrast', 0.5)])
        
        return round(total_score, 1)
    
    def _update_results(self, category):
        """Atualiza estatísticas de resultado"""
        if category in self.results:
            self.results[category] += 1
    
    def _generate_processing_report(self, output_folder):
        """Gera relatório de processamento"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'results': self.results,
            'config_used': self.config,
            'success_rate': ((self.results['total'] - self.results['failures']) / self.results['total'] * 100) if self.results['total'] > 0 else 0
        }
        
        report_path = os.path.join(output_folder, 'processing_report.json')
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Relatório salvo em {report_path}")
        
        # Print summary
        logger.info("\n📈 RESUMO DO PROCESSAMENTO:")
        logger.info("=" * 40)
        logger.info(f"Total processadas: {self.results['total']}")
        logger.info(f"✅ Selecionadas: {self.results['selected']}")
        logger.info(f"💫 Desfocadas: {self.results['blurry']}")
        logger.info(f"🌑 Iluminação: {self.results['low_light']}")
        logger.info(f"🔄 Duplicadas: {self.results['duplicates']}")
        logger.info(f"❌ Falhas: {self.results['failures']}")
        logger.info(f"Taxa de sucesso: {report['success_rate']:.1f}%")
    
    def _get_blur_threshold_by_strategy(self, strategy):
        """Get blur threshold based on configured strategy"""
        return get_threshold_by_strategy(strategy)

def process_images_with_ai(input_folder, output_folder, ai_classifier=None):
    """
    Função de conveniência para processar imagens com IA
    
    Args:
        input_folder: Pasta de entrada
        output_folder: Pasta de saída
        ai_classifier: Classificador IA (opcional)
        
    Returns:
        dict: Estatísticas do processamento
    """
    processor = ImageProcessor()
    
    # Load AI classifier if not provided
    if ai_classifier is None:
        try:
            from .ai_classifier import AIClassifier
            ai_classifier = AIClassifier()
            ai_classifier.load_best_model()
        except:
            logger.warning("Classificador IA não disponível, usando regras básicas")
            ai_classifier = None
    
    return processor.process_folder(
        input_folder, 
        output_folder, 
        use_ai=(ai_classifier is not None),
        ai_classifier=ai_classifier
    )

def process_images_basic(input_folder, output_folder):
    """
    Função de conveniência para processamento básico (sem IA)
    
    Args:
        input_folder: Pasta de entrada
        output_folder: Pasta de saída
        
    Returns:
        dict: Estatísticas do processamento
    """
    processor = ImageProcessor()
    return processor.process_folder(input_folder, output_folder, use_ai=False)

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Uso: python image_processor.py <pasta_entrada> <pasta_saida> [--ai]")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    use_ai = '--ai' in sys.argv
    
    if use_ai:
        print("🤖 Processamento com IA habilitado")
        results = process_images_with_ai(input_dir, output_dir)
    else:
        print("📏 Processamento com regras básicas")
        results = process_images_basic(input_dir, output_dir)
    
    print(f"✅ Processamento concluído: {results}")
