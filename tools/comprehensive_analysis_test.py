#!/usr/bin/env python3
"""
Comprehensive Photo Analysis Test - 100 Random Images
Teste abrangente de análise de fotos com 100 imagens aleatórias

This script performs complete analysis on 100 random images from the dataset,
generating detailed visualizations and statistics for system validation.
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
    print("🚀 Sistema otimizado para análise em lote")
    
except Exception as e:
    print(f"⚠️ Otimização não disponível: {e}")

# Import analysis modules
from src.core.feature_extractor import FeatureExtractor
from src.core.person_detector import PersonDetector


class ComprehensiveImageAnalyzer:
    """
    Comprehensive image analyzer for system validation
    Analisador abrangente de imagens para validação do sistema
    """
    
    def __init__(self, sample_size: int = 100):
        """Initialize analyzer with specified sample size"""
        self.sample_size = sample_size
        self.extractor = FeatureExtractor()
        self.person_detector = PersonDetector()
        
        # Create output directory
        self.output_dir = Path("data/quality/comprehensive_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results = []
        self.statistics = {}
        
        print(f"🔬 Analisador inicializado para {sample_size} imagens")
    
    def select_random_images(self, input_dir: str) -> List[str]:
        """Select random images from input directory"""
        input_path = Path(input_dir)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        all_images = [
            str(img) for img in input_path.rglob("*") 
            if img.suffix.lower() in image_extensions
        ]
        
        if len(all_images) < self.sample_size:
            print(f"⚠️ Apenas {len(all_images)} imagens disponíveis, usando todas")
            return all_images
        
        # Random selection
        selected = random.sample(all_images, self.sample_size)
        print(f"✅ {len(selected)} imagens selecionadas aleatoriamente")
        
        return selected
    
    def analyze_single_image(self, image_path: str) -> Dict[str, Any]:
        """Perform complete analysis on single image"""
        try:
            # Extract features
            features = self.extractor.extract_features(image_path)
            
            # Load image for visualization
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
            
            # Detect persons for visualization
            persons = self.person_detector.detect_persons_and_faces(image)
            
            # Compile analysis result
            result = {
                'filename': Path(image_path).name,
                'path': image_path,
                'features': features,
                'persons': persons if isinstance(persons, list) else [persons] if persons else [],
                'image_shape': image.shape,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Erro ao analisar {Path(image_path).name}: {e}")
            return {
                'filename': Path(image_path).name,
                'path': image_path,
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def create_image_visualization(self, result: Dict[str, Any], save_path: str):
        """Create comprehensive visualization for a single image"""
        try:
            if 'error' in result:
                return
            
            # Load image
            image = cv2.imread(result['path'])
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f"Análise Completa: {result['filename']}", fontsize=16, fontweight='bold')
            
            # 1. Original image with person detection
            ax1 = axes[0, 0]
            ax1.imshow(image_rgb)
            ax1.set_title("Imagem Original + Detecção de Pessoas")
            ax1.axis('off')
            
            # Draw person bboxes
            if result['persons']:
                for i, person in enumerate(result['persons']):
                    if isinstance(person, dict) and 'bbox' in person:
                        x, y, w, h = person['bbox']
                        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                               edgecolor='red', facecolor='none', alpha=0.8)
                        ax1.add_patch(rect)
                        ax1.text(x, y-10, f'Pessoa {i+1}', color='red', fontweight='bold')
            
            # 2. Technical Quality Metrics
            ax2 = axes[0, 1]
            features = result['features']
            
            technical_metrics = {
                'Blur Score': features.get('sharpness_laplacian', 0),
                'Brightness': features.get('brightness_mean', 0),
                'Contrast': features.get('contrast_std', 0),
                'Final Score': features.get('final_score', 0) * 100 if features.get('final_score') else 0
            }
            
            bars = ax2.bar(technical_metrics.keys(), technical_metrics.values(), 
                          color=['blue', 'orange', 'green', 'purple'])
            ax2.set_title("Métricas Técnicas")
            ax2.set_ylabel("Valores")
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, technical_metrics.values()):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.1f}', ha='center', va='bottom')
            
            # 3. Person Analysis Summary
            ax3 = axes[0, 2]
            person_data = {
                'Pessoas': features.get('person_count', 0),
                'Faces': features.get('face_count', 0),
                'Pessoa Dominante': 1 if features.get('dominant_person_score', 0) > 0 else 0,
                'Cortes Detectados': 1 if features.get('cropping_severity') != 'none' else 0
            }
            
            colors = ['skyblue', 'lightgreen', 'gold', 'salmon']
            bars3 = ax3.bar(person_data.keys(), person_data.values(), color=colors)
            ax3.set_title("Análise de Pessoas")
            ax3.set_ylabel("Contagem")
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
            
            # Add value labels
            for bar, value in zip(bars3, person_data.values()):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{int(value)}', ha='center', va='bottom')
            
            # 4. Exposure Analysis
            ax4 = axes[1, 0]
            exposure_level = features.get('exposure_level', 'unknown')
            exposure_score = features.get('exposure_quality_score', 0)
            
            # Create pie chart for exposure
            exposure_colors = {
                'adequate': 'green',
                'dark': 'darkblue',
                'bright': 'yellow',
                'extremely_dark': 'black',
                'extremely_bright': 'white'
            }
            
            color = exposure_colors.get(exposure_level, 'gray')
            ax4.pie([exposure_score, 1-exposure_score], 
                   labels=[f'{exposure_level.title()}\n({exposure_score:.2f})', 'Deficiência'],
                   colors=[color, 'lightgray'], autopct='%1.1f%%')
            ax4.set_title("Análise de Exposição")
            
            # 5. Quality Issues Summary
            ax5 = axes[1, 1]
            quality_issues = []
            
            # Check for various issues
            if features.get('overexposure_is_critical', False):
                quality_issues.append('Superexposição Crítica')
            if features.get('cropping_severity') in ['moderate', 'severe']:
                quality_issues.append('Corte Problemático')
            if features.get('sharpness_laplacian', 100) < 50:
                quality_issues.append('Imagem Borrada')
            if features.get('pose_naturalness_score', 1) < 0.5:
                quality_issues.append('Pose Não Natural')
            if not quality_issues:
                quality_issues.append('Sem Problemas Detectados')
            
            # Create text summary
            ax5.text(0.1, 0.9, "Problemas Detectados:", fontsize=12, fontweight='bold', 
                    transform=ax5.transAxes)
            
            for i, issue in enumerate(quality_issues[:5]):  # Max 5 issues
                color = 'red' if issue != 'Sem Problemas Detectados' else 'green'
                ax5.text(0.1, 0.8 - i*0.1, f"• {issue}", fontsize=10, 
                        color=color, transform=ax5.transAxes)
            
            ax5.set_xlim(0, 1)
            ax5.set_ylim(0, 1)
            ax5.axis('off')
            ax5.set_title("Problemas de Qualidade")
            
            # 6. Recommendations
            ax6 = axes[1, 2]
            rating = features.get('rating', 'N/A')
            recommendation = features.get('recommendation', 'N/A')
            final_score = features.get('final_score', 0) * 100 if features.get('final_score') else 0
            
            # Create recommendation display
            rec_color = {
                'excellent': 'darkgreen',
                'good': 'green', 
                'fair': 'orange',
                'poor': 'red',
                'reject': 'darkred'
            }.get(rating, 'gray')
            
            ax6.text(0.5, 0.8, f"Rating: {rating.upper()}", fontsize=14, fontweight='bold',
                    ha='center', va='center', transform=ax6.transAxes, color=rec_color)
            
            ax6.text(0.5, 0.6, f"Score: {final_score:.1f}%", fontsize=12,
                    ha='center', va='center', transform=ax6.transAxes)
            
            ax6.text(0.5, 0.4, f"Recomendação:", fontsize=10, fontweight='bold',
                    ha='center', va='center', transform=ax6.transAxes)
            
            ax6.text(0.5, 0.3, f"{recommendation}", fontsize=10,
                    ha='center', va='center', transform=ax6.transAxes, style='italic')
            
            ax6.set_xlim(0, 1)
            ax6.set_ylim(0, 1)
            ax6.axis('off')
            ax6.set_title("Classificação Final")
            
            # Save visualization
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"❌ Erro ao criar visualização para {result['filename']}: {e}")
    
    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics"""
        if not self.results:
            return
        
        # Extract features for analysis
        valid_results = [r for r in self.results if 'features' in r]
        
        if not valid_results:
            print("❌ Nenhum resultado válido para análise estatística")
            return
        
        # Compile statistics
        stats = {
            'total_images': len(self.results),
            'successful_analysis': len(valid_results),
            'failed_analysis': len(self.results) - len(valid_results),
            'success_rate': len(valid_results) / len(self.results) * 100
        }
        
        # Feature statistics
        features_data = []
        for result in valid_results:
            features = result['features']
            features_data.append({
                'filename': result['filename'],
                'blur_score': features.get('sharpness_laplacian', 0),
                'brightness': features.get('brightness_mean', 0),
                'person_count': features.get('person_count', 0),
                'final_score': features.get('final_score', 0) * 100 if features.get('final_score') else 0,
                'rating': features.get('rating', 'unknown'),
                'exposure_level': features.get('exposure_level', 'unknown'),
                'has_overexposure': features.get('overexposure_is_critical', False),
                'cropping_issues': features.get('cropping_severity', 'none') != 'none'
            })
        
        df = pd.DataFrame(features_data)
        
        # Calculate detailed statistics
        stats.update({
            'blur_stats': {
                'mean': df['blur_score'].mean(),
                'std': df['blur_score'].std(),
                'min': df['blur_score'].min(),
                'max': df['blur_score'].max(),
                'blurry_count': (df['blur_score'] < 50).sum()
            },
            'brightness_stats': {
                'mean': df['brightness'].mean(),
                'std': df['brightness'].std(),
                'dark_count': (df['brightness'] < 80).sum(),
                'bright_count': (df['brightness'] > 180).sum()
            },
            'person_stats': {
                'mean_persons': df['person_count'].mean(),
                'images_with_people': (df['person_count'] > 0).sum(),
                'multi_person_images': (df['person_count'] > 1).sum()
            },
            'quality_stats': {
                'mean_score': df['final_score'].mean(),
                'high_quality': (df['final_score'] > 75).sum(),
                'low_quality': (df['final_score'] < 25).sum(),
                'rating_distribution': df['rating'].value_counts().to_dict(),
                'exposure_distribution': df['exposure_level'].value_counts().to_dict()
            },
            'problem_stats': {
                'overexposure_critical': df['has_overexposure'].sum(),
                'cropping_issues': df['cropping_issues'].sum()
            }
        })
        
        self.statistics = stats
        
        # Save statistics to JSON
        stats_file = self.output_dir / "analysis_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"📊 Estatísticas salvas em: {stats_file}")
        
        return stats
    
    def create_summary_visualizations(self):
        """Create summary visualizations and charts"""
        if not self.statistics:
            return
        
        # Create summary plots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f"Análise Resumo - {self.sample_size} Imagens Aleatórias", 
                    fontsize=16, fontweight='bold')
        
        # 1. Success Rate
        ax1 = axes[0, 0]
        success_data = [self.statistics['successful_analysis'], self.statistics['failed_analysis']]
        ax1.pie(success_data, labels=['Sucesso', 'Falha'], autopct='%1.1f%%', 
               colors=['green', 'red'])
        ax1.set_title(f"Taxa de Sucesso: {self.statistics['success_rate']:.1f}%")
        
        # 2. Quality Distribution
        ax2 = axes[0, 1]
        if 'rating_distribution' in self.statistics['quality_stats']:
            rating_dist = self.statistics['quality_stats']['rating_distribution']
            colors = ['darkgreen', 'green', 'orange', 'red', 'darkred'][:len(rating_dist)]
            ax2.bar(rating_dist.keys(), rating_dist.values(), color=colors)
            ax2.set_title("Distribuição de Qualidade")
            ax2.set_ylabel("Quantidade")
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Person Detection
        ax3 = axes[0, 2]
        person_stats = self.statistics['person_stats']
        person_data = [
            person_stats['images_with_people'],
            self.statistics['successful_analysis'] - person_stats['images_with_people']
        ]
        ax3.pie(person_data, labels=['Com Pessoas', 'Sem Pessoas'], autopct='%1.1f%%',
               colors=['lightblue', 'lightgray'])
        ax3.set_title(f"Detecção de Pessoas\nMédia: {person_stats['mean_persons']:.1f}/imagem")
        
        # 4. Blur Analysis
        ax4 = axes[1, 0]
        blur_stats = self.statistics['blur_stats']
        blur_data = [
            blur_stats['blurry_count'],
            self.statistics['successful_analysis'] - blur_stats['blurry_count']
        ]
        ax4.pie(blur_data, labels=['Borradas', 'Nítidas'], autopct='%1.1f%%',
               colors=['red', 'green'])
        ax4.set_title(f"Análise de Blur\nMédia: {blur_stats['mean']:.1f}")
        
        # 5. Exposure Analysis
        ax5 = axes[1, 1]
        if 'exposure_distribution' in self.statistics['quality_stats']:
            exp_dist = self.statistics['quality_stats']['exposure_distribution']
            exp_colors = ['green', 'darkblue', 'yellow', 'black', 'orange'][:len(exp_dist)]
            ax5.bar(exp_dist.keys(), exp_dist.values(), color=exp_colors)
            ax5.set_title("Distribuição de Exposição")
            ax5.set_ylabel("Quantidade")
            plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
        
        # 6. Problems Summary
        ax6 = axes[1, 2]
        problem_stats = self.statistics['problem_stats']
        problems = {
            'Superexposição': problem_stats['overexposure_critical'],
            'Cortes': problem_stats['cropping_issues'],
            'Blur': self.statistics['blur_stats']['blurry_count']
        }
        
        ax6.bar(problems.keys(), problems.values(), color=['orange', 'red', 'purple'])
        ax6.set_title("Problemas Detectados")
        ax6.set_ylabel("Quantidade")
        
        # Add value labels
        for i, (key, value) in enumerate(problems.items()):
            ax6.text(i, value + 0.5, str(value), ha='center', va='bottom')
        
        # Save summary visualization
        plt.tight_layout()
        summary_path = self.output_dir / "analysis_summary.png"
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Visualização resumo salva em: {summary_path}")
    
    def run_comprehensive_analysis(self, input_dir: str = "data/input"):
        """Run complete analysis on random sample"""
        print(f"\n🔬 INICIANDO ANÁLISE ABRANGENTE - {self.sample_size} IMAGENS")
        print("=" * 80)
        
        # Select random images
        image_paths = self.select_random_images(input_dir)
        
        if not image_paths:
            print("❌ Nenhuma imagem encontrada para análise")
            return
        
        # Analyze each image
        print(f"\n📊 Analisando {len(image_paths)} imagens...")
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"   [{i:3d}/{len(image_paths)}] {Path(image_path).name}")
            
            # Analyze image
            result = self.analyze_single_image(image_path)
            self.results.append(result)
            
            # Create individual visualization for first 10 images
            if i <= 10 and 'features' in result:
                viz_path = self.output_dir / f"analysis_{i:03d}_{Path(image_path).stem}.png"
                self.create_image_visualization(result, str(viz_path))
        
        # Generate statistics
        print(f"\n📊 Gerando estatísticas...")
        self.generate_summary_statistics()
        
        # Create summary visualizations
        print(f"📈 Criando visualizações resumo...")
        self.create_summary_visualizations()
        
        # Print summary
        self.print_analysis_summary()
        
        print(f"\n✅ Análise completa! Resultados salvos em: {self.output_dir}")
    
    def print_analysis_summary(self):
        """Print detailed analysis summary"""
        if not self.statistics:
            return
        
        print(f"\n📋 RESUMO DA ANÁLISE")
        print("=" * 80)
        
        stats = self.statistics
        
        print(f"🎯 Análise Geral:")
        print(f"   • Total de imagens: {stats['total_images']}")
        print(f"   • Análises bem-sucedidas: {stats['successful_analysis']}")
        print(f"   • Taxa de sucesso: {stats['success_rate']:.1f}%")
        
        if 'blur_stats' in stats:
            blur = stats['blur_stats']
            print(f"\n🔍 Análise de Blur:")
            print(f"   • Score médio: {blur['mean']:.1f}")
            print(f"   • Imagens borradas: {blur['blurry_count']}")
            print(f"   • Faixa: {blur['min']:.1f} - {blur['max']:.1f}")
        
        if 'person_stats' in stats:
            person = stats['person_stats']
            print(f"\n👥 Análise de Pessoas:")
            print(f"   • Pessoas por imagem (média): {person['mean_persons']:.1f}")
            print(f"   • Imagens com pessoas: {person['images_with_people']}")
            print(f"   • Imagens multi-pessoa: {person['multi_person_images']}")
        
        if 'quality_stats' in stats:
            quality = stats['quality_stats']
            print(f"\n⭐ Análise de Qualidade:")
            print(f"   • Score médio: {quality['mean_score']:.1f}%")
            print(f"   • Alta qualidade (>75%): {quality['high_quality']}")
            print(f"   • Baixa qualidade (<25%): {quality['low_quality']}")
        
        if 'problem_stats' in stats:
            problems = stats['problem_stats']
            print(f"\n⚠️ Problemas Detectados:")
            print(f"   • Superexposição crítica: {problems['overexposure_critical']}")
            print(f"   • Problemas de corte: {problems['cropping_issues']}")


def main():
    """Main execution function"""
    print("🎯 TESTE ABRANGENTE - PHOTO CULLING SYSTEM v2.5")
    print("Análise de 100 imagens aleatórias com visualizações detalhadas")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = ComprehensiveImageAnalyzer(sample_size=100)
    
    # Run comprehensive analysis
    analyzer.run_comprehensive_analysis()
    
    print(f"\n🎉 ANÁLISE CONCLUÍDA!")
    print(f"📁 Todos os resultados estão em: {analyzer.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
