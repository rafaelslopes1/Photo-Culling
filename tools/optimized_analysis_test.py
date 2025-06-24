#!/usr/bin/env python3
"""
Optimized Photo Analysis Test - 20 Random Images with Detailed Visualization
Teste otimizado de análise de fotos com 20 imagens aleatórias e visualizações detalhadas

This script performs complete analysis on 20 random images from the dataset,
generating detailed visualizations and statistics for system validation.
Optimized for fast execution and comprehensive visual feedback.
"""

import os
import sys
import cv2
import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import seaborn as sns
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Setup optimization and logging
try:
    from src.utils.gpu_optimizer import MacM3Optimizer
    from src.utils.logging_config import enable_quiet_mode
    
    # Enable quiet mode first
    enable_quiet_mode()
    
    # Setup GPU optimization
    gpu_config, system_info = MacM3Optimizer.setup_quiet_and_optimized()
    print("🚀 Sistema otimizado para análise rápida")
    
except Exception as e:
    print(f"⚠️ Otimização não disponível: {e}")

# Import analysis modules
from src.core.feature_extractor import FeatureExtractor
from src.core.person_detector import PersonDetector


class OptimizedImageAnalyzer:
    """
    Optimized image analyzer for quick system validation
    Analisador otimizado de imagens para validação rápida do sistema
    """
    
    def __init__(self, sample_size: int = 20):
        """Initialize analyzer with specified sample size"""
        self.sample_size = sample_size
        self.extractor = FeatureExtractor()
        self.person_detector = PersonDetector()
        
        # Create output directory
        self.output_dir = Path("data/quality/optimized_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = []
        self.statistics = {}
        
        print(f"🔬 Analisador otimizado inicializado para {sample_size} imagens")
    
    def select_random_images(self, input_dir: str) -> List[str]:
        """Select random images from input directory"""
        input_path = Path(input_dir)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        all_images = [
            str(img) for img in input_path.rglob("*") 
            if img.suffix.lower() in image_extensions and img.stat().st_size > 10000  # Min 10KB
        ]
        
        if len(all_images) < self.sample_size:
            print(f"⚠️ Apenas {len(all_images)} imagens disponíveis, usando todas")
            return all_images
        
        # Random selection
        selected = random.sample(all_images, self.sample_size)
        print(f"✅ {len(selected)} imagens selecionadas aleatoriamente")
        
        return selected
    
    def analyze_single_image_optimized(self, image_path: str) -> Dict[str, Any]:
        """Perform optimized analysis on single image"""
        try:
            # Load image first for basic info
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
            
            # Extract basic features (skip expensive operations)
            features = {}
            
            # Basic image metrics
            height, width = image.shape[:2]
            features['image_width'] = width
            features['image_height'] = height
            features['image_area'] = width * height
            
            # Quick blur detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            features['sharpness_laplacian'] = blur_score
            
            # Basic brightness
            brightness = np.mean(gray)
            features['brightness_mean'] = brightness
            
            # Quick contrast
            contrast = np.std(gray)
            features['contrast_std'] = contrast
            
            # Simple exposure check
            dark_pixels = np.sum(gray < 50) / gray.size
            bright_pixels = np.sum(gray > 200) / gray.size
            
            if dark_pixels > 0.3:
                exposure_level = 'dark'
            elif bright_pixels > 0.2:
                exposure_level = 'bright'
            else:
                exposure_level = 'adequate'
            
            features['exposure_level'] = exposure_level
            features['dark_pixel_ratio'] = dark_pixels
            features['bright_pixel_ratio'] = bright_pixels
            
            # Quick person detection and face detection
            try:
                persons_data = self.person_detector.detect_persons_and_faces(image)
                
                # Extract persons and faces separately
                if isinstance(persons_data, dict):
                    persons = persons_data.get('persons', [])
                    faces = persons_data.get('faces', [])
                    
                    features['person_count'] = len(persons)
                    features['face_count'] = len(faces)
                elif isinstance(persons_data, list):
                    # Legacy format - assume persons
                    features['person_count'] = len(persons_data)
                    features['face_count'] = sum(1 for p in persons_data if isinstance(p, dict) and p.get('face_bbox'))
                else:
                    features['person_count'] = 1 if persons_data else 0
                    features['face_count'] = 1 if persons_data and isinstance(persons_data, dict) and persons_data.get('face_bbox') else 0
                    
            except Exception as e:
                print(f"   ⚠️ Detecção de pessoas falhou: {e}")
                features['person_count'] = 0
                features['face_count'] = 0
                persons_data = {}
            
            # Quick quality assessment
            quality_score = 0
            
            # Blur component (0-40 points)
            if blur_score > 100:
                quality_score += 40
            elif blur_score > 50:
                quality_score += 25
            else:
                quality_score += 10
            
            # Brightness component (0-30 points)
            if 80 <= brightness <= 180:
                quality_score += 30
            elif 60 <= brightness <= 200:
                quality_score += 20
            else:
                quality_score += 10
            
            # Person detection bonus (0-30 points)
            if features['person_count'] > 0:
                quality_score += 20
                if features['face_count'] > 0:
                    quality_score += 10
            
            features['final_score'] = quality_score / 100.0
            
            # Rating based on score
            if quality_score >= 80:
                rating = 'excellent'
            elif quality_score >= 60:
                rating = 'good'
            elif quality_score >= 40:
                rating = 'fair'
            elif quality_score >= 20:
                rating = 'poor'
            else:
                rating = 'reject'
            
            features['rating'] = rating
            
            # Simple recommendation
            if rating in ['excellent', 'good']:
                recommendation = 'Manter - boa qualidade'
            elif rating == 'fair':
                recommendation = 'Revisar - qualidade moderada'
            else:
                recommendation = 'Rejeitar - baixa qualidade'
            
            features['recommendation'] = recommendation
            
            # Compile result
            result = {
                'filename': Path(image_path).name,
                'path': image_path,
                'features': features,
                'persons': persons_data.get('persons', []) if isinstance(persons_data, dict) else [],
                'image_shape': image.shape,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            print(f"   ❌ Erro ao analisar {Path(image_path).name}: {e}")
            return {
                'filename': Path(image_path).name,
                'path': image_path,
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def create_detailed_visualization(self, result: Dict[str, Any], save_path: str):
        """Create detailed visualization for a single image"""
        try:
            if 'error' in result:
                return
            
            # Load image
            image = cv2.imread(result['path'])
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f"Análise Detalhada: {result['filename']}", fontsize=16, fontweight='bold')
            
            # 1. Original image with annotations
            ax1 = axes[0, 0]
            ax1.imshow(image_rgb)
            ax1.set_title("Imagem Original + Detecções")
            ax1.axis('off')
            
            # Draw person bboxes if available
            if result['persons']:
                for i, person in enumerate(result['persons']):
                    if isinstance(person, dict) and 'bbox' in person:
                        x, y, w, h = person['bbox']
                        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                               edgecolor='red', facecolor='none', alpha=0.8)
                        ax1.add_patch(rect)
                        ax1.text(x, y-10, f'P{i+1}', color='red', fontweight='bold', fontsize=12)
                        
                        # Draw face bbox if available
                        if 'face_bbox' in person:
                            fx, fy, fw, fh = person['face_bbox']
                            face_rect = patches.Rectangle((fx, fy), fw, fh, linewidth=2,
                                                        edgecolor='blue', facecolor='none', alpha=0.8)
                            ax1.add_patch(face_rect)
                            ax1.text(fx, fy-10, 'Face', color='blue', fontweight='bold', fontsize=10)
            
            # 2. Quality Metrics
            ax2 = axes[0, 1]
            features = result['features']
            
            metrics = {
                'Blur': min(features.get('sharpness_laplacian', 0), 200),
                'Brilho': features.get('brightness_mean', 0),
                'Contraste': min(features.get('contrast_std', 0), 100),
                'Score Final': features.get('final_score', 0) * 100
            }
            
            colors = ['red' if metrics['Blur'] < 50 else 'green',
                     'orange' if metrics['Brilho'] < 80 or metrics['Brilho'] > 180 else 'green',
                     'blue', 'purple']
            
            bars = ax2.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.7)
            ax2.set_title("Métricas de Qualidade")
            ax2.set_ylabel("Valores")
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            for bar, value in zip(bars, metrics.values()):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{value:.1f}', ha='center', va='bottom', fontsize=10)
                        
            # 3. Detection Summary
            ax3 = axes[0, 2]
            detection_data = {
                'Pessoas': features.get('person_count', 0),
                'Faces': features.get('face_count', 0)
            }
            
            bars3 = ax3.bar(detection_data.keys(), detection_data.values(), 
                          color=['lightblue', 'lightgreen'], alpha=0.8)
            ax3.set_title("Detecções")
            ax3.set_ylabel("Quantidade")
            ax3.set_ylim(0, max(max(detection_data.values()), 1) + 1)
            
            for bar, value in zip(bars3, detection_data.values()):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(value)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            # 4. Exposure Analysis
            ax4 = axes[1, 0]
            exposure = features.get('exposure_level', 'unknown')
            dark_ratio = features.get('dark_pixel_ratio', 0) * 100
            bright_ratio = features.get('bright_pixel_ratio', 0) * 100
            
            exposure_data = [dark_ratio, 100 - dark_ratio - bright_ratio, bright_ratio]
            labels = ['Escuro', 'Normal', 'Claro']
            colors_exp = ['darkblue', 'green', 'yellow']
            
            wedges, texts, autotexts = ax4.pie(exposure_data, labels=labels, colors=colors_exp, 
                                             autopct='%1.1f%%', startangle=90)
            ax4.set_title(f"Exposição: {exposure.title()}")
            
            # 5. Image Information
            ax5 = axes[1, 1]
            info_text = f"""
Arquivo: {result['filename']}
Resolução: {features.get('image_width', 0)} × {features.get('image_height', 0)}
Área: {features.get('image_area', 0):,} pixels
Exposição: {exposure.title()}
Pessoas: {features.get('person_count', 0)}
Faces: {features.get('face_count', 0)}
            """.strip()
            
            ax5.text(0.1, 0.9, info_text, transform=ax5.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace')
            ax5.set_xlim(0, 1)
            ax5.set_ylim(0, 1)
            ax5.axis('off')
            ax5.set_title("Informações da Imagem")
            
            # 6. Final Assessment
            ax6 = axes[1, 2]
            rating = features.get('rating', 'unknown')
            score = features.get('final_score', 0) * 100
            recommendation = features.get('recommendation', 'N/A')
            
            # Color based on rating
            rating_colors = {
                'excellent': 'darkgreen',
                'good': 'green',
                'fair': 'orange', 
                'poor': 'red',
                'reject': 'darkred'
            }
            
            color = rating_colors.get(rating, 'gray')
            
            # Create score circle
            from matplotlib.patches import Circle
            circle = Circle((0.5, 0.7), 0.2, color=color, alpha=0.3)
            ax6.add_patch(circle)
            
            ax6.text(0.5, 0.7, f'{score:.0f}%', ha='center', va='center', 
                    fontsize=20, fontweight='bold', transform=ax6.transAxes)
            
            ax6.text(0.5, 0.5, rating.upper(), ha='center', va='center',
                    fontsize=14, fontweight='bold', color=color, transform=ax6.transAxes)
            
            ax6.text(0.5, 0.3, recommendation, ha='center', va='center',
                    fontsize=10, style='italic', transform=ax6.transAxes, wrap=True)
            
            ax6.set_xlim(0, 1)
            ax6.set_ylim(0, 1)
            ax6.axis('off')
            ax6.set_title("Avaliação Final")
            
            # Save visualization
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"   ❌ Erro na visualização de {result['filename']}: {e}")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        if not self.results:
            return
        
        valid_results = [r for r in self.results if 'features' in r]
        
        if not valid_results:
            print("❌ Nenhum resultado válido para relatório")
            return
        
        # Calculate statistics
        stats = {
            'analysis_summary': {
                'total_images': len(self.results),
                'successful_analysis': len(valid_results),
                'failed_analysis': len(self.results) - len(valid_results),
                'success_rate': len(valid_results) / len(self.results) * 100
            }
        }
        
        # Feature analysis
        features_data = []
        for result in valid_results:
            features = result['features']
            features_data.append({
                'filename': result['filename'],
                'blur_score': features.get('sharpness_laplacian', 0),
                'brightness': features.get('brightness_mean', 0),
                'contrast': features.get('contrast_std', 0),
                'person_count': features.get('person_count', 0),
                'face_count': features.get('face_count', 0),
                'final_score': features.get('final_score', 0) * 100,
                'rating': features.get('rating', 'unknown'),
                'exposure_level': features.get('exposure_level', 'unknown')
            })
        
        df = pd.DataFrame(features_data)
        
        # Detailed statistics
        stats.update({
            'quality_analysis': {
                'blur_statistics': {
                    'mean_blur_score': df['blur_score'].mean(),
                    'blur_threshold_50': (df['blur_score'] < 50).sum(),
                    'sharp_images': (df['blur_score'] >= 50).sum()
                },
                'brightness_analysis': {
                    'mean_brightness': df['brightness'].mean(),
                    'dark_images': (df['brightness'] < 80).sum(),
                    'bright_images': (df['brightness'] > 180).sum(),
                    'normal_exposure': ((df['brightness'] >= 80) & (df['brightness'] <= 180)).sum()
                },
                'detection_analysis': {
                    'images_with_people': (df['person_count'] > 0).sum(),
                    'images_with_faces': (df['face_count'] > 0).sum(),
                    'average_people_per_image': df['person_count'].mean(),
                    'multi_person_images': (df['person_count'] > 1).sum()
                },
                'overall_quality': {
                    'mean_score': df['final_score'].mean(),
                    'excellent_images': (df['rating'] == 'excellent').sum(),
                    'good_images': (df['rating'] == 'good').sum(),
                    'fair_images': (df['rating'] == 'fair').sum(),
                    'poor_images': (df['rating'] == 'poor').sum(),
                    'reject_images': (df['rating'] == 'reject').sum()
                }
            }
        })
        
        self.statistics = stats
        
        # Save to JSON
        report_file = self.output_dir / "analysis_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        
        # Save detailed CSV
        csv_file = self.output_dir / "detailed_results.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        return stats
    
    def create_summary_dashboard(self):
        """Create comprehensive summary dashboard"""
        if not self.statistics:
            return
        
        # Create dashboard
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f"Dashboard de Análise - {self.sample_size} Imagens Aleatórias", 
                    fontsize=18, fontweight='bold')
        
        stats = self.statistics
        
        # 1. Success Rate
        ax1 = axes[0, 0]
        success_data = [stats['analysis_summary']['successful_analysis'], 
                       stats['analysis_summary']['failed_analysis']]
        ax1.pie(success_data, labels=['Sucesso', 'Falha'], autopct='%1.1f%%', 
               colors=['green', 'red'], explode=(0.05, 0))
        ax1.set_title(f"Taxa de Sucesso\n{stats['analysis_summary']['success_rate']:.1f}%")
        
        # 2. Quality Distribution
        ax2 = axes[0, 1]
        quality_stats = stats['quality_analysis']['overall_quality']
        quality_data = [
            quality_stats['excellent_images'],
            quality_stats['good_images'],
            quality_stats['fair_images'],
            quality_stats['poor_images'],
            quality_stats['reject_images']
        ]
        quality_labels = ['Excelente', 'Bom', 'Razoável', 'Ruim', 'Rejeitar']
        quality_colors = ['darkgreen', 'green', 'orange', 'red', 'darkred']
        
        bars2 = ax2.bar(quality_labels, quality_data, color=quality_colors, alpha=0.8)
        ax2.set_title(f"Distribuição de Qualidade\nScore Médio: {quality_stats['mean_score']:.1f}%")
        ax2.set_ylabel("Quantidade")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        for bar, value in zip(bars2, quality_data):
            if value > 0:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(value)}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Detection Analysis
        ax3 = axes[0, 2]
        detection_stats = stats['quality_analysis']['detection_analysis']
        detection_data = [
            detection_stats['images_with_people'],
            stats['analysis_summary']['successful_analysis'] - detection_stats['images_with_people']
        ]
        ax3.pie(detection_data, labels=['Com Pessoas', 'Sem Pessoas'], autopct='%1.1f%%',
               colors=['lightblue', 'lightgray'], explode=(0.05, 0))
        ax3.set_title(f"Detecção de Pessoas\nMédia: {detection_stats['average_people_per_image']:.1f}/img")
        
        # 4. Blur Analysis
        ax4 = axes[1, 0]
        blur_stats = stats['quality_analysis']['blur_statistics']
        blur_data = [blur_stats['sharp_images'], blur_stats['blur_threshold_50']]
        ax4.pie(blur_data, labels=['Nítidas', 'Borradas'], autopct='%1.1f%%',
               colors=['green', 'red'], explode=(0.05, 0))
        ax4.set_title(f"Análise de Nitidez\nScore Médio: {blur_stats['mean_blur_score']:.1f}")
        
        # 5. Brightness Analysis
        ax5 = axes[1, 1]
        brightness_stats = stats['quality_analysis']['brightness_analysis']
        brightness_data = [
            brightness_stats['dark_images'],
            brightness_stats['normal_exposure'],
            brightness_stats['bright_images']
        ]
        brightness_labels = ['Escuras', 'Normais', 'Claras']
        brightness_colors = ['darkblue', 'green', 'yellow']
        
        bars5 = ax5.bar(brightness_labels, brightness_data, color=brightness_colors, alpha=0.8)
        ax5.set_title(f"Análise de Exposição\nBrilho Médio: {brightness_stats['mean_brightness']:.1f}")
        ax5.set_ylabel("Quantidade")
        
        for bar, value in zip(bars5, brightness_data):
            if value > 0:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(value)}', ha='center', va='bottom', fontweight='bold')
        
        # 6. System Performance Summary
        ax6 = axes[1, 2]
        performance_text = f"""
RESUMO DA ANÁLISE

✅ Taxa de Sucesso: {stats['analysis_summary']['success_rate']:.1f}%
📊 Imagens Analisadas: {stats['analysis_summary']['successful_analysis']}
⭐ Score Médio: {quality_stats['mean_score']:.1f}%

🔍 DETECÇÕES:
👥 Com Pessoas: {detection_stats['images_with_people']}
👤 Com Faces: {detection_stats['images_with_faces']}
🎯 Multi-pessoa: {detection_stats['multi_person_images']}

📈 QUALIDADE:
🟢 Excelente/Bom: {quality_stats['excellent_images'] + quality_stats['good_images']}
🟡 Razoável: {quality_stats['fair_images']}
🔴 Ruim/Rejeitar: {quality_stats['poor_images'] + quality_stats['reject_images']}
        """.strip()
        
        ax6.text(0.05, 0.95, performance_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace')
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title("Resumo do Sistema")
        
        # Save dashboard
        plt.tight_layout()
        dashboard_path = self.output_dir / "analysis_dashboard.png"
        plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Dashboard salvo em: {dashboard_path}")
    
    def run_optimized_analysis(self, input_dir: str = "data/input"):
        """Run optimized analysis on sample"""
        print(f"\n🚀 ANÁLISE OTIMIZADA - {self.sample_size} IMAGENS ALEATÓRIAS")
        print("=" * 80)
        
        # Select images
        image_paths = self.select_random_images(input_dir)
        
        if not image_paths:
            print("❌ Nenhuma imagem encontrada")
            return
        
        print(f"\n📊 Processando {len(image_paths)} imagens...")
        
        # Process each image
        for i, image_path in enumerate(image_paths, 1):
            print(f" [{i:2d}/{len(image_paths)}] {Path(image_path).name}")
            
            # Analyze
            result = self.analyze_single_image_optimized(image_path)
            self.results.append(result)
            
            # Create visualization for each image
            if 'features' in result:
                viz_path = self.output_dir / f"detailed_{i:02d}_{Path(image_path).stem}.png"
                self.create_detailed_visualization(result, str(viz_path))
        
        # Generate reports
        print(f"\n📊 Gerando relatórios...")
        self.generate_summary_report()
        self.create_summary_dashboard()
        
        # Print summary
        self.print_final_summary()
        
        print(f"\n✅ ANÁLISE CONCLUÍDA!")
        print(f"📁 Resultados salvos em: {self.output_dir}")
        print("=" * 80)
    
    def print_final_summary(self):
        """Print final analysis summary"""
        if not self.statistics:
            return
        
        print(f"\n📋 RESUMO FINAL DA ANÁLISE")
        print("=" * 80)
        
        stats = self.statistics
        analysis = stats['analysis_summary']  
        quality = stats['quality_analysis']
        
        print(f"🎯 RESULTADOS GERAIS:")
        print(f"   • Total analisado: {analysis['successful_analysis']}/{analysis['total_images']}")
        print(f"   • Taxa de sucesso: {analysis['success_rate']:.1f}%")
        
        print(f"\n⭐ QUALIDADE:")
        overall = quality['overall_quality']
        print(f"   • Score médio: {overall['mean_score']:.1f}%")
        print(f"   • Excelente: {overall['excellent_images']} | Bom: {overall['good_images']}")
        print(f"   • Razoável: {overall['fair_images']} | Ruim: {overall['poor_images']} | Rejeitar: {overall['reject_images']}")
        
        print(f"\n🔍 DETECÇÕES:")
        detection = quality['detection_analysis']
        print(f"   • Imagens com pessoas: {detection['images_with_people']}")
        print(f"   • Imagens com faces: {detection['images_with_faces']}")
        print(f"   • Pessoas por imagem (média): {detection['average_people_per_image']:.1f}")
        
        print(f"\n📊 ANÁLISE TÉCNICA:")
        blur = quality['blur_statistics']
        brightness = quality['brightness_analysis']
        print(f"   • Blur score médio: {blur['mean_blur_score']:.1f}")
        print(f"   • Imagens nítidas: {blur['sharp_images']} | Borradas: {blur['blur_threshold_50']}")
        print(f"   • Brilho médio: {brightness['mean_brightness']:.1f}")
        print(f"   • Exposição normal: {brightness['normal_exposure']} | Escuras: {brightness['dark_images']} | Claras: {brightness['bright_images']}")


def main():
    """Main execution function"""
    print("🎯 TESTE OTIMIZADO - PHOTO CULLING SYSTEM v2.5")
    print("Análise rápida com visualizações detalhadas de cada imagem")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = OptimizedImageAnalyzer(sample_size=20)
    
    # Run analysis
    analyzer.run_optimized_analysis()


if __name__ == "__main__":
    main()
