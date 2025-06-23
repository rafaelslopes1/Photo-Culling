#!/usr/bin/env python3
"""
Image Quality Analyzer for Photo Culling System
Analisador de qualidade de imagem - detecção de blur, foco e outros problemas
Versão otimizada com thresholds baseados em validação supervisionada
"""

import cv2
import numpy as np
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sqlite3
from datetime import datetime

# Import configuração consolidada
try:
    import sys
    import os
    # Add data/quality to path for blur_config import
    data_quality_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'quality')
    if data_quality_path not in sys.path:
        sys.path.insert(0, data_quality_path)
    
    from blur_config import (
        MODERATE_PRACTICAL as DEFAULT_THRESHOLD,
        get_blur_threshold,
        classify_blur_level
    )
    HAS_OPTIMIZED_CONFIG = True
except ImportError as e:
    # Fallback para valores padrão se não encontrar a configuração
    DEFAULT_THRESHOLD = 60
    HAS_OPTIMIZED_CONFIG = False
    
    def get_blur_threshold(strategy='balanced'):
        thresholds = {'conservative': 50, 'balanced': 60, 'aggressive': 100}
        return thresholds.get(strategy, 60)
    
    def classify_blur_level(score, strategy='balanced'):
        threshold = get_blur_threshold(strategy)
        if score < threshold:
            return {'level': 'blurry', 'recommendation': 'review'}
        else:
            return {'level': 'sharp', 'recommendation': 'keep'}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageQualityAnalyzer:
    """
    Analisador de qualidade de imagem com foco em detecção de blur/desfoque
    Implementa o método Variance of Laplacian para medição de foco
    """
    
    def __init__(self, 
                 blur_threshold: float = DEFAULT_THRESHOLD,
                 results_db: str = "../data/quality/quality_analysis.db"):
        """
        Inicializa o analisador de qualidade
        
        Args:
            blur_threshold: Limiar para detecção de blur (padrão: 100.0)
            results_db: Banco de dados para armazenar resultados
        """
        self.blur_threshold = blur_threshold
        self.results_db = results_db
        self._setup_database()
        
        logger.info(f"🔍 Image Quality Analyzer initialized")
        logger.info(f"   📊 Blur threshold: {blur_threshold}")
        logger.info(f"   💾 Results DB: {results_db}")
    
    def _setup_database(self):
        """Configura banco de dados para armazenar resultados de qualidade"""
        os.makedirs(os.path.dirname(self.results_db), exist_ok=True)
        
        conn = sqlite3.connect(self.results_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quality_analysis (
                filename TEXT PRIMARY KEY,
                blur_score REAL NOT NULL,
                is_blurry BOOLEAN NOT NULL,
                analysis_timestamp TEXT NOT NULL,
                image_size TEXT,
                mean_brightness REAL,
                contrast_score REAL,
                sharpness_score REAL,
                noise_level REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Quality analysis database initialized")
    
    def variance_of_laplacian(self, image: np.ndarray) -> float:
        """
        Calcula a variância do Laplaciano para medir o foco da imagem
        
        Método baseado em Pech-Pacheco et al. (2000):
        "Diatom autofocusing in brightfield microscopy: a comparative study"
        
        Args:
            image: Imagem em escala de cinza
            
        Returns:
            float: Medida de foco (valores maiores = mais foco)
        """
        # Verifica se a imagem está em escala de cinza
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Aplica o operador Laplaciano e calcula a variância
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return laplacian.var()
    
    def analyze_single_image(self, image_path: str) -> Dict:
        """
        Analisa uma única imagem para detectar problemas de qualidade
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            Dict: Resultados da análise completa
        """
        try:
            # Carrega a imagem
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Converte para escala de cinza
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Análise de foco/blur
            blur_score = self.variance_of_laplacian(gray)
            is_blurry = blur_score < self.blur_threshold
            
            # Análises adicionais de qualidade
            image_size = f"{image.shape[1]}x{image.shape[0]}"  # width x height
            mean_brightness = np.mean(gray)
            contrast_score = np.std(gray)  # Desvio padrão como medida de contraste
            
            # Sharpness usando gradiente Sobel
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sharpness_score = np.mean(np.sqrt(sobel_x**2 + sobel_y**2))
            
            # Estimativa de ruído usando diferença da média local
            kernel = np.ones((3,3), np.float32) / 9
            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            noise_level = np.mean(np.abs(gray.astype(np.float32) - local_mean))
            
            results = {
                'filename': os.path.basename(image_path),
                'filepath': image_path,
                'blur_score': round(blur_score, 2),
                'is_blurry': is_blurry,
                'blur_status': 'BLURRY' if is_blurry else 'SHARP',
                'image_size': image_size,
                'mean_brightness': round(mean_brightness, 2),
                'contrast_score': round(contrast_score, 2),
                'sharpness_score': round(sharpness_score, 2),
                'noise_level': round(noise_level, 2),
                'analysis_timestamp': datetime.now().isoformat(),
                'quality_rating': self._calculate_quality_rating(
                    float(blur_score), float(mean_brightness), float(contrast_score), 
                    float(sharpness_score), float(noise_level)
                )
            }
            
            # Salva no banco de dados
            self._save_analysis_result(results)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {image_path}: {e}")
            return {
                'filename': os.path.basename(image_path),
                'filepath': image_path,
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def _calculate_quality_rating(self, blur_score: float, brightness: float, 
                                contrast: float, sharpness: float, noise: float) -> str:
        """
        Calcula uma classificação geral de qualidade da imagem
        
        Returns:
            str: EXCELLENT, GOOD, FAIR, POOR
        """
        score = 0
        
        # Pontuação baseada no foco/blur (peso: 40%)
        if blur_score > 500:
            score += 4
        elif blur_score > 200:
            score += 3
        elif blur_score > 100:
            score += 2
        elif blur_score > 50:
            score += 1
        
        # Pontuação baseada no contraste (peso: 25%)
        if contrast > 60:
            score += 2.5
        elif contrast > 40:
            score += 2
        elif contrast > 20:
            score += 1
        
        # Pontuação baseada no brilho (peso: 20%)
        if 60 <= brightness <= 180:  # Bom range de brilho
            score += 2
        elif 40 <= brightness <= 220:
            score += 1.5
        elif 20 <= brightness <= 240:
            score += 1
        
        # Pontuação baseada no ruído (peso: 15%) - menos ruído é melhor
        if noise < 10:
            score += 1.5
        elif noise < 20:
            score += 1
        elif noise < 30:
            score += 0.5
        
        # Converte pontuação para rating
        if score >= 8:
            return "EXCELLENT"
        elif score >= 6:
            return "GOOD"
        elif score >= 4:
            return "FAIR"
        else:
            return "POOR"
    
    def _save_analysis_result(self, results: Dict):
        """Salva resultado da análise no banco de dados"""
        if 'error' in results:
            return  # Não salva resultados com erro
        
        conn = sqlite3.connect(self.results_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO quality_analysis 
            (filename, blur_score, is_blurry, analysis_timestamp, 
             image_size, mean_brightness, contrast_score, sharpness_score, noise_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            results['filename'],
            results['blur_score'],
            results['is_blurry'],
            results['analysis_timestamp'],
            results['image_size'],
            results['mean_brightness'],
            results['contrast_score'],
            results['sharpness_score'],
            results['noise_level']
        ))
        
        conn.commit()
        conn.close()
    
    def analyze_folder(self, folder_path: str, 
                      extensions: Optional[List[str]] = None) -> Dict:
        """
        Analisa todas as imagens em uma pasta
        
        Args:
            folder_path: Caminho para a pasta
            extensions: Lista de extensões (default: comum image formats)
            
        Returns:
            Dict: Estatísticas da análise
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        folder_path_obj = Path(folder_path)
        
        if not folder_path_obj.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        # Encontra todas as imagens
        image_files = []
        for ext in extensions:
            image_files.extend(folder_path_obj.glob(f"*{ext}"))
            image_files.extend(folder_path_obj.glob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.warning(f"⚠️ No images found in {folder_path}")
            return {'total_images': 0}
        
        logger.info(f"🔍 Starting quality analysis of {len(image_files)} images...")
        
        results = []
        stats = {
            'total_images': len(image_files),
            'analyzed': 0,
            'errors': 0,
            'blurry_images': 0,
            'sharp_images': 0,
            'quality_distribution': {'EXCELLENT': 0, 'GOOD': 0, 'FAIR': 0, 'POOR': 0}
        }
        
        for i, image_file in enumerate(image_files, 1):
            logger.info(f"📷 Analyzing ({i}/{len(image_files)}): {image_file.name}")
            
            result = self.analyze_single_image(str(image_file))
            results.append(result)
            
            if 'error' in result:
                stats['errors'] += 1
            else:
                stats['analyzed'] += 1
                if result['is_blurry']:
                    stats['blurry_images'] += 1
                else:
                    stats['sharp_images'] += 1
                
                # Contabiliza rating de qualidade
                quality_rating = result.get('quality_rating', 'POOR')
                stats['quality_distribution'][quality_rating] += 1
        
        # Calcula estatísticas finais
        if stats['analyzed'] > 0:
            stats['blur_percentage'] = round(
                (stats['blurry_images'] / stats['analyzed']) * 100, 1
            )
            
            # Scores médios
            blur_scores = [r['blur_score'] for r in results if 'blur_score' in r]
            if blur_scores:
                stats['avg_blur_score'] = round(np.mean(blur_scores), 2)
                stats['min_blur_score'] = round(min(blur_scores), 2)
                stats['max_blur_score'] = round(max(blur_scores), 2)
        
        stats['results'] = results
        stats['analysis_timestamp'] = datetime.now().isoformat()
        
        logger.info("🎉 Quality analysis completed!")
        logger.info(f"   📊 Total: {stats['total_images']} images")
        logger.info(f"   ✅ Analyzed: {stats['analyzed']}")
        logger.info(f"   ❌ Errors: {stats['errors']}")
        logger.info(f"   🌫️ Blurry: {stats['blurry_images']} ({stats.get('blur_percentage', 0)}%)")
        logger.info(f"   🔎 Sharp: {stats['sharp_images']}")
        
        return stats
    
    def get_blurry_images(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Retorna lista de imagens detectadas como borradas
        
        Args:
            limit: Limite de resultados (None = todos)
            
        Returns:
            List[Dict]: Lista de imagens borradas
        """
        conn = sqlite3.connect(self.results_db)
        cursor = conn.cursor()
        
        query = '''
            SELECT filename, blur_score, analysis_timestamp, image_size,
                   mean_brightness, contrast_score, sharpness_score
            FROM quality_analysis 
            WHERE is_blurry = 1 
            ORDER BY blur_score ASC
        '''
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            results.append({
                'filename': row[0],
                'blur_score': row[1],
                'analysis_timestamp': row[2],
                'image_size': row[3],
                'mean_brightness': row[4],
                'contrast_score': row[5],
                'sharpness_score': row[6]
            })
        
        return results
    
    def get_quality_report(self) -> Dict:
        """
        Gera relatório completo de qualidade das imagens analisadas
        
        Returns:
            Dict: Relatório detalhado
        """
        conn = sqlite3.connect(self.results_db)
        cursor = conn.cursor()
        
        # Estatísticas gerais
        cursor.execute('SELECT COUNT(*) FROM quality_analysis')
        total_analyzed = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM quality_analysis WHERE is_blurry = 1')
        total_blurry = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(blur_score) FROM quality_analysis')
        avg_blur_score = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT AVG(mean_brightness) FROM quality_analysis')
        avg_brightness = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT AVG(contrast_score) FROM quality_analysis')
        avg_contrast = cursor.fetchone()[0] or 0
        
        # Top 10 piores imagens (mais borradas)
        cursor.execute('''
            SELECT filename, blur_score FROM quality_analysis 
            WHERE is_blurry = 1 
            ORDER BY blur_score ASC LIMIT 10
        ''')
        worst_images = cursor.fetchall()
        
        # Top 10 melhores imagens (mais nítidas)
        cursor.execute('''
            SELECT filename, blur_score FROM quality_analysis 
            WHERE is_blurry = 0 
            ORDER BY blur_score DESC LIMIT 10
        ''')
        best_images = cursor.fetchall()
        
        conn.close()
        
        report = {
            'summary': {
                'total_analyzed': total_analyzed,
                'total_blurry': total_blurry,
                'total_sharp': total_analyzed - total_blurry,
                'blur_percentage': round((total_blurry / total_analyzed * 100), 1) if total_analyzed > 0 else 0,
                'avg_blur_score': round(avg_blur_score, 2),
                'avg_brightness': round(avg_brightness, 2),
                'avg_contrast': round(avg_contrast, 2)
            },
            'worst_images': [{'filename': row[0], 'blur_score': row[1]} for row in worst_images],
            'best_images': [{'filename': row[0], 'blur_score': row[1]} for row in best_images],
            'blur_threshold': self.blur_threshold,
            'report_timestamp': datetime.now().isoformat()
        }
        
        return report

    def analyze_with_optimized_threshold(self, image_path: str, 
                                       threshold_mode: str = 'balanced') -> Dict:
        """
        Analisa imagem usando thresholds otimizados por validação supervisionada
        
        Args:
            image_path: Caminho para a imagem
            threshold_mode: 'conservative', 'balanced', 'aggressive', 'very_aggressive'
        
        Returns:
            Dict: Análise completa com categorização otimizada
        """
        # Análise básica
        result = self.analyze_single_image(image_path)
        
        if 'blur_score' in result:
            # Usa configuração consolidada
            categorization = classify_blur_level(result['blur_score'], threshold_mode)
            
            # Adiciona informações da classificação
            result.update({
                'threshold_mode': threshold_mode,
                'threshold_used': get_blur_threshold(threshold_mode),
                'blur_category': categorization['level'],
                'recommended_action': categorization['action'],
                'action_description': categorization['description'],
                'confidence': categorization['confidence'],
                'optimized_analysis': True
            })
            
            # Log simplificado baseado na categorização
            if categorization['action'] == 'remove':
                logger.info(f"🗑️ REMOVER: {result['filename']} - {categorization['description']}")
            elif categorization['action'] == 'review':
                logger.info(f"👁️ REVISAR: {result['filename']} - {categorization['description']}")
            else:
                logger.info(f"✅ MANTER: {result['filename']} - {categorization['description']}")
        
        return result
    
    def batch_analysis_with_strategy(self, folder_path: str, 
                                   strategy: str = 'general') -> Dict:
        """
        Análise em lote usando estratégia específica
        
        Args:
            folder_path: Pasta com imagens
            strategy: 'archive', 'general', 'professional', 'exhibition'
        
        Returns:
            Dict: Estatísticas e recomendações
        """
        # Map strategy to threshold mode
        strategy_map = {
            'archive': 'conservative',
            'general': 'balanced', 
            'professional': 'aggressive',
            'exhibition': 'aggressive'
        }
        threshold_mode = strategy_map.get(strategy, 'balanced')
        threshold_value = get_blur_threshold(threshold_mode)
        
        logger.info(f"🎯 Análise em lote - Estratégia: {strategy}")
        logger.info(f"📊 Threshold: {threshold_value} ({threshold_mode})")
        
        # Executa análise folder normal primeiro
        stats = self.analyze_folder(folder_path)
        
        if stats['total_images'] == 0:
            return stats
        
        # Re-categoriza usando thresholds otimizados
        categorized_results = {
            'extremely_blurry': [],
            'very_blurry': [],
            'blurry': [],
            'acceptable': [],
            'sharp': []
        }
        
        for result in stats['results']:
            if 'blur_score' in result:
                categorization = classify_blur_level(result['blur_score'], threshold_mode)
                category = categorization['level']
                categorized_results[category].append({
                    'filename': result['filename'],
                    'blur_score': result['blur_score'],
                    'action': categorization['recommendation']
                })
        
        # Calcula estatísticas otimizadas
        total_analyzed = sum(len(cat) for cat in categorized_results.values())
        categories_to_remove = ['extremely_blurry', 'blurry']  # Categories that should be removed
        to_remove = sum(len(categorized_results[cat]) for cat in categories_to_remove if cat in categorized_results)
        to_keep = total_analyzed - to_remove
        
        optimized_stats = {
            **stats,  # Mantém estatísticas originais
            'strategy_used': strategy,
            'threshold_mode': threshold_mode,
            'threshold_value': threshold_value,
            'categorized_results': categorized_results,
            'optimization_summary': {
                'total_analyzed': total_analyzed,
                'recommended_to_remove': to_remove,
                'recommended_to_keep': to_keep,
                'removal_percentage': round((to_remove / total_analyzed * 100), 1) if total_analyzed > 0 else 0,
                'categories_for_removal': categories_to_remove
            }
        }
        
        # Log resumo
        logger.info(f"📈 RESUMO DA ESTRATÉGIA '{strategy.upper()}':")
        logger.info(f"   Total analisadas: {total_analyzed}")
        logger.info(f"   🗑️ Para remover: {to_remove} ({optimized_stats['optimization_summary']['removal_percentage']}%)")
        logger.info(f"   ✅ Para manter: {to_keep}")
        
        for category, items in categorized_results.items():
            if items:
                emoji = {'extremely_blurry': '🔴', 'very_blurry': '🟡', 'blurry': '🟠', 
                        'acceptable': '🟢', 'sharp': '🔵'}.get(category, '⚪')
                logger.info(f"   {emoji} {category}: {len(items)}")
        
        return optimized_stats

def main():
    """Função principal para teste do módulo"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Image Quality Analyzer - Blur Detection')
    parser.add_argument('--input', '-i', required=True, 
                       help='Path to image file or folder')
    parser.add_argument('--threshold', '-t', type=float, default=100.0,
                       help='Blur threshold (default: 100.0)')
    parser.add_argument('--report', '-r', action='store_true',
                       help='Generate quality report')
    parser.add_argument('--list-blurry', '-b', action='store_true',
                       help='List all blurry images')
    
    args = parser.parse_args()
    
    # Inicializa analisador
    analyzer = ImageQualityAnalyzer(blur_threshold=args.threshold)
    
    if args.report:
        # Gera relatório
        report = analyzer.get_quality_report()
        print("\n" + "="*60)
        print("📊 QUALITY ANALYSIS REPORT")
        print("="*60)
        print(f"Total analyzed: {report['summary']['total_analyzed']}")
        print(f"Blurry images: {report['summary']['total_blurry']} ({report['summary']['blur_percentage']}%)")
        print(f"Sharp images: {report['summary']['total_sharp']}")
        print(f"Average blur score: {report['summary']['avg_blur_score']}")
        print(f"Blur threshold: {report['blur_threshold']}")
        
        if report['worst_images']:
            print(f"\n🌫️ TOP 10 BLURRIEST IMAGES:")
            for img in report['worst_images'][:10]:
                print(f"   {img['filename']}: {img['blur_score']:.2f}")
        
        if report['best_images']:
            print(f"\n🔎 TOP 10 SHARPEST IMAGES:")
            for img in report['best_images'][:10]:
                print(f"   {img['filename']}: {img['blur_score']:.2f}")
    
    elif args.list_blurry:
        # Lista imagens borradas
        blurry_images = analyzer.get_blurry_images()
        print(f"\n🌫️ BLURRY IMAGES DETECTED ({len(blurry_images)} total):")
        print("-" * 60)
        for img in blurry_images:
            print(f"{img['filename']}: {img['blur_score']:.2f} (analyzed: {img['analysis_timestamp'][:19]})")
    
    else:
        # Analisa input
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Analisa arquivo único
            print(f"\n🔍 Analyzing single image: {input_path.name}")
            result = analyzer.analyze_single_image(str(input_path))
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"📊 Results:")
                print(f"   Blur Score: {result['blur_score']:.2f}")
                print(f"   Status: {result['blur_status']}")
                print(f"   Quality Rating: {result['quality_rating']}")
                print(f"   Brightness: {result['mean_brightness']:.2f}")
                print(f"   Contrast: {result['contrast_score']:.2f}")
                print(f"   Sharpness: {result['sharpness_score']:.2f}")
                
        elif input_path.is_dir():
            # Analisa pasta
            print(f"\n🔍 Analyzing folder: {input_path}")
            stats = analyzer.analyze_folder(str(input_path))
            
            print(f"\n📈 ANALYSIS COMPLETE:")
            print(f"   Total images: {stats['total_images']}")
            print(f"   Successfully analyzed: {stats['analyzed']}")
            print(f"   Errors: {stats['errors']}")
            print(f"   Blurry images: {stats['blurry_images']} ({stats.get('blur_percentage', 0)}%)")
            print(f"   Sharp images: {stats['sharp_images']}")
            
            if 'avg_blur_score' in stats:
                print(f"   Average blur score: {stats['avg_blur_score']}")
                print(f"   Range: {stats['min_blur_score']} - {stats['max_blur_score']}")
        
        else:
            print(f"❌ Input path not found: {input_path}")

if __name__ == "__main__":
    main()
