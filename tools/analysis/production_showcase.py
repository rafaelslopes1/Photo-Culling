#!/usr/bin/env python3
"""
Production Showcase - Demonstração Completa do Sistema
Ferramenta para teste de produção com análise completa, visualizações e relatórios
"""

import sys
import os
import json
import random
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.feature_extractor import FeatureExtractor
from src.core.person_detector import PersonDetector
from src.core.exposure_analyzer import ExposureAnalyzer


def convert_numpy_types(obj):
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Configure logging to be less verbose
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class ProductionShowcase:
    """
    Sistema de demonstração completa para produção
    """
    
    def __init__(self, output_dir: str = "data/analysis_results/production_showcase"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.person_detector = PersonDetector()
        self.exposure_analyzer = ExposureAnalyzer()
        
        # Results storage
        self.results = {}
        self.analysis_summary = {}
        
    def select_sample_images(self, input_dir: str, count: int = 10) -> List[Path]:
        """
        Seleciona imagens de amostra de forma inteligente
        """
        input_path = Path(input_dir)
        all_images = list(input_path.glob("*.JPG")) + list(input_path.glob("*.jpg"))
        
        if len(all_images) < count:
            print(f"⚠️ Apenas {len(all_images)} imagens disponíveis (solicitado: {count})")
            return all_images
            
        # Seleção estratificada: algumas do início, meio e fim para variedade
        selected = []
        step = len(all_images) // count
        
        for i in range(count):
            idx = i * step + random.randint(0, min(step-1, len(all_images)-1-i*step))
            if idx < len(all_images):
                selected.append(all_images[idx])
        
        # Garantir exatamente 'count' imagens únicas
        selected = list(set(selected))
        while len(selected) < count and len(selected) < len(all_images):
            remaining = [img for img in all_images if img not in selected]
            selected.append(random.choice(remaining))
            
        return selected[:count]
    
    def analyze_image_complete(self, image_path: Path) -> Dict[str, Any]:
        """
        Análise completa de uma imagem
        """
        print(f"🔍 Analisando: {image_path.name}")
        
        try:
            # Extract all features
            features = self.feature_extractor.extract_features(str(image_path))
            
            # Load image for visualization
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
            
            # Person detection with detailed info
            person_results = self.person_detector.detect_persons_and_faces(image)
            
            # Exposure analysis
            exposure_result = self.exposure_analyzer.analyze_exposure(image)
            
            # Compile complete analysis
            analysis = {
                'filename': image_path.name,
                'path': str(image_path),
                'timestamp': datetime.now().isoformat(),
                'image_dimensions': {
                    'height': image.shape[0],
                    'width': image.shape[1],
                    'channels': image.shape[2]
                },
                'features': features,
                'person_detection': {
                    'persons_detected': len(person_results.get('persons', [])) if person_results else 0,
                    'faces_detected': len(person_results.get('faces', [])) if person_results else 0,
                    'persons': []
                },
                'exposure_analysis': exposure_result,
                'quality_assessment': self._assess_quality(features),
                'recommendations': self._generate_recommendations(features)
            }
            
            # Add detailed person info
            if person_results and person_results.get('persons'):
                for i, person in enumerate(person_results.get('persons', [])):
                    person_info = {
                        'person_id': i,
                        'bbox': person.bounding_box,
                        'confidence': person.confidence,
                        'landmarks_count': len(person.landmarks) if person.landmarks else 0,
                        'dominance_score': person.dominance_score
                    }
                    analysis['person_detection']['persons'].append(person_info)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Erro ao analisar {image_path.name}: {e}")
            return {
                'filename': image_path.name,
                'path': str(image_path),
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _assess_quality(self, features: Dict) -> Dict[str, Any]:
        """
        Avaliação de qualidade baseada nas features
        """
        sharpness = features.get('sharpness_laplacian', 0)
        exposure = features.get('exposure_level', 'unknown')
        person_count = features.get('person_count', 0)
        dominant_score = features.get('dominant_person_score', 0)
        
        # Quality scoring
        quality_score = 0.0
        quality_factors = []
        
        # Sharpness assessment
        if sharpness > 100:
            quality_score += 0.3
            quality_factors.append("sharp")
        elif sharpness > 50:
            quality_score += 0.2
            quality_factors.append("moderately_sharp")
        else:
            quality_factors.append("blurry")
            
        # Exposure assessment
        if exposure in ['adequate', 'bright']:
            quality_score += 0.3
            quality_factors.append("good_exposure")
        elif exposure in ['dark', 'extremely_bright']:
            quality_score += 0.1
            quality_factors.append("exposure_issues")
        else:
            quality_factors.append("poor_exposure")
            
        # Person detection assessment
        if person_count > 0:
            quality_score += 0.2
            quality_factors.append("persons_detected")
            if dominant_score > 0.3:
                quality_score += 0.2
                quality_factors.append("strong_dominant_person")
        else:
            quality_factors.append("no_persons")
            
        # Final rating
        if quality_score >= 0.8:
            rating = "excellent"
        elif quality_score >= 0.6:
            rating = "good"
        elif quality_score >= 0.4:
            rating = "fair"
        elif quality_score >= 0.2:
            rating = "poor"
        else:
            rating = "reject"
            
        return {
            'overall_score': round(quality_score, 3),
            'rating': rating,
            'quality_factors': quality_factors,
            'technical_metrics': {
                'sharpness_score': sharpness,
                'exposure_level': exposure,
                'person_count': person_count,
                'dominant_person_score': round(dominant_score, 3)
            }
        }
    
    def _generate_recommendations(self, features: Dict) -> List[str]:
        """
        Gera recomendações baseadas na análise
        """
        recommendations = []
        
        sharpness = features.get('sharpness_laplacian', 0)
        exposure = features.get('exposure_level', 'unknown')
        person_count = features.get('person_count', 0)
        
        if sharpness < 50:
            recommendations.append("⚠️ Imagem com blur detectado - considerar rejeição para uso profissional")
        elif sharpness > 150:
            recommendations.append("✅ Excelente nitidez - ótima para impressão e uso profissional")
            
        if exposure == 'extremely_dark':
            recommendations.append("🌑 Imagem muito escura - requererá pós-processamento significativo")
        elif exposure == 'extremely_bright':
            recommendations.append("☀️ Imagem superexposta - informações podem estar perdidas")
        elif exposure == 'adequate':
            recommendations.append("✅ Exposição adequada - pronta para uso")
            
        if person_count == 0:
            recommendations.append("👤 Nenhuma pessoa detectada - verificar se é intencional")
        elif person_count > 3:
            recommendations.append("👥 Múltiplas pessoas detectadas - verificar enquadramento")
            
        if not recommendations:
            recommendations.append("✅ Imagem em boas condições gerais")
            
        return recommendations
    
    def create_annotated_visualization(self, image_path: Path, analysis: Dict) -> Optional[Path]:
        """
        Cria visualização anotada da imagem
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Não foi possível carregar: {image_path}")
            
            # Create a copy for annotation
            annotated = image.copy()
            
            # Get person detection info
            persons_info = analysis.get('person_detection', {}).get('persons', [])
            
            # Draw person bounding boxes
            for i, person in enumerate(persons_info):
                bbox = person.get('bbox', [])
                if len(bbox) == 4:
                    x, y, w, h = bbox
                    
                    # Draw bounding box
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw person label
                    label = f"Person {i+1} (Score: {person.get('dominance_score', 0):.2f})"
                    cv2.putText(annotated, label, (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add analysis info overlay
            overlay_info = [
                f"Arquivo: {analysis['filename']}",
                f"Rating: {analysis['quality_assessment']['rating'].upper()}",
                f"Nitidez: {analysis['quality_assessment']['technical_metrics']['sharpness_score']:.1f}",
                f"Exposição: {analysis['quality_assessment']['technical_metrics']['exposure_level']}",
                f"Pessoas: {analysis['quality_assessment']['technical_metrics']['person_count']}"
            ]
            
            # Draw info box
            box_height = 25 * len(overlay_info) + 20
            cv2.rectangle(annotated, (10, 10), (400, box_height), (0, 0, 0), -1)
            cv2.rectangle(annotated, (10, 10), (400, box_height), (255, 255, 255), 2)
            
            for i, info in enumerate(overlay_info):
                y_pos = 35 + i * 25
                cv2.putText(annotated, info, (20, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Save annotated image
            output_path = self.output_dir / f"annotated_{analysis['filename']}"
            cv2.imwrite(str(output_path), annotated)
            
            return output_path
            
        except Exception as e:
            logger.error(f"Erro ao criar visualização para {image_path.name}: {e}")
            return None
    
    def generate_natural_language_report(self, analysis: Dict) -> str:
        """
        Gera relatório em linguagem natural para uma imagem
        """
        filename = analysis['filename']
        quality = analysis['quality_assessment']
        features = analysis['features']
        persons = analysis['person_detection']
        exposure = analysis['exposure_analysis']
        
        report = f"\n📸 **ANÁLISE: {filename}**\n"
        report += "=" * 50 + "\n"
        
        # Overall assessment
        rating = quality['rating']
        score = quality['overall_score']
        
        rating_emoji = {
            'excellent': '🌟',
            'good': '✅', 
            'fair': '⚖️',
            'poor': '⚠️',
            'reject': '❌'
        }
        
        report += f"{rating_emoji.get(rating, '❓')} **CLASSIFICAÇÃO: {rating.upper()}** (Score: {score:.1f}/1.0)\n\n"
        
        # Technical analysis
        sharpness = quality['technical_metrics']['sharpness_score']
        exposure_level = quality['technical_metrics']['exposure_level']
        person_count = quality['technical_metrics']['person_count']
        
        report += "🔍 **ANÁLISE TÉCNICA:**\n"
        report += f"  • Nitidez: {sharpness:.1f} - "
        if sharpness > 100:
            report += "Excelente qualidade de foco\n"
        elif sharpness > 50:
            report += "Foco adequado\n"
        else:
            report += "Imagem com blur significativo\n"
            
        report += f"  • Exposição: {exposure_level} - "
        exposure_desc = {
            'extremely_dark': 'Muito escura, requer correção',
            'dark': 'Levemente escura',
            'adequate': 'Exposição ideal',
            'bright': 'Levemente clara', 
            'extremely_bright': 'Superexposta, pode ter perdas'
        }
        report += exposure_desc.get(exposure_level, 'Nível desconhecido') + "\n"
        
        # Person analysis
        report += f"  • Pessoas detectadas: {person_count}\n"
        
        if person_count > 0:
            dominant_score = quality['technical_metrics']['dominant_person_score']
            report += f"  • Score da pessoa dominante: {dominant_score:.2f}\n"
            
            if dominant_score > 0.4:
                report += "    → Pessoa muito bem posicionada e focada\n"
            elif dominant_score > 0.2:
                report += "    → Pessoa adequadamente posicionada\n"
            else:
                report += "    → Pessoa pode estar mal enquadrada ou desfocada\n"
        
        # Detailed insights
        report += "\n💡 **INSIGHTS DETALHADOS:**\n"
        
        # Quality factors analysis
        factors = quality['quality_factors']
        positive_factors = [f for f in factors if f in ['sharp', 'good_exposure', 'persons_detected', 'strong_dominant_person']]
        negative_factors = [f for f in factors if f in ['blurry', 'poor_exposure', 'no_persons', 'exposure_issues']]
        
        if positive_factors:
            report += "  ✅ Pontos fortes:\n"
            for factor in positive_factors:
                factor_desc = {
                    'sharp': 'Imagem nítida e bem focada',
                    'good_exposure': 'Exposição bem balanceada', 
                    'persons_detected': 'Pessoas identificadas com sucesso',
                    'strong_dominant_person': 'Pessoa principal bem destacada'
                }
                report += f"    • {factor_desc.get(factor, factor)}\n"
        
        if negative_factors:
            report += "  ⚠️ Pontos de atenção:\n"
            for factor in negative_factors:
                factor_desc = {
                    'blurry': 'Falta de nitidez pode comprometer qualidade',
                    'poor_exposure': 'Problemas de exposição requerem correção',
                    'no_persons': 'Nenhuma pessoa identificada na imagem',
                    'exposure_issues': 'Exposição não ideal mas recuperável'
                }
                report += f"    • {factor_desc.get(factor, factor)}\n"
        
        # Recommendations
        recommendations = analysis['recommendations']
        if recommendations:
            report += "\n🎯 **RECOMENDAÇÕES:**\n"
            for rec in recommendations:
                report += f"  {rec}\n"
        
        # Final verdict
        report += "\n📋 **VEREDICTO FINAL:**\n"
        if rating == 'excellent':
            report += "  🌟 Imagem de excelente qualidade, ideal para qualquer uso profissional.\n"
        elif rating == 'good':
            report += "  ✅ Boa qualidade geral, adequada para a maioria dos usos.\n"
        elif rating == 'fair':
            report += "  ⚖️ Qualidade aceitável, mas pode requerer edição menor.\n"
        elif rating == 'poor':
            report += "  ⚠️ Qualidade comprometida, usar apenas se necessário.\n"
        else:
            report += "  ❌ Qualidade inadequada, recomenda-se rejeição.\n"
        
        return report
    
    def run_production_showcase(self, input_dir: str = "data/input", sample_count: int = 10):
        """
        Executa demonstração completa de produção
        """
        print("🚀 INICIANDO DEMONSTRAÇÃO DE PRODUÇÃO")
        print("=" * 60)
        
        # Select sample images
        print(f"📂 Selecionando {sample_count} imagens de amostra...")
        selected_images = self.select_sample_images(input_dir, sample_count)
        print(f"✅ {len(selected_images)} imagens selecionadas")
        
        # Process each image
        all_analyses = []
        all_reports = []
        
        print(f"\n🔍 Processando {len(selected_images)} imagens...")
        for i, image_path in enumerate(selected_images, 1):
            print(f"\n[{i}/{len(selected_images)}] {image_path.name}")
            
            # Complete analysis
            analysis = self.analyze_image_complete(image_path)
            all_analyses.append(analysis)
            
            # Create visualization
            if 'error' not in analysis:
                viz_path = self.create_annotated_visualization(image_path, analysis)
                if viz_path:
                    analysis['visualization_path'] = str(viz_path)
                    print(f"  📊 Visualização salva: {viz_path.name}")
                
                # Generate natural language report
                report = self.generate_natural_language_report(analysis)
                all_reports.append(report)
                print(f"  📝 Relatório gerado")
            else:
                print(f"  ❌ Erro na análise: {analysis.get('error', 'Desconhecido')}")
        
        # JSON results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"production_showcase_results_{timestamp}.json"
        
        # Convert numpy types before JSON serialization
        json_data = convert_numpy_types({
            'timestamp': datetime.now().isoformat(),
            'sample_count': len(selected_images),
            'successful_analyses': len([a for a in all_analyses if 'error' not in a]),
            'analyses': all_analyses,
            'summary_statistics': self._generate_summary_stats(all_analyses)
        })
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # Natural language report
        report_path = self.output_dir / f"production_showcase_report_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 📊 RELATÓRIO DE DEMONSTRAÇÃO - PHOTO CULLING SYSTEM\n")
            f.write(f"**Data:** {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}\n")
            f.write(f"**Amostras analisadas:** {len(selected_images)}\n\n")
            
            for report in all_reports:
                f.write(report + "\n")
            
            # Summary
            f.write(self._generate_executive_summary(all_analyses))
        
        print(f"\n🎉 DEMONSTRAÇÃO CONCLUÍDA!")
        print(f"📄 Resultados JSON: {json_path}")
        print(f"📝 Relatório completo: {report_path}")
        print(f"📊 Visualizações: {self.output_dir}")
        
        return {
            'json_path': json_path,
            'report_path': report_path,
            'analyses': all_analyses,
            'visualizations_dir': self.output_dir
        }
    
    def _generate_summary_stats(self, analyses: List[Dict]) -> Dict[str, Any]:
        """
        Gera estatísticas resumidas
        """
        successful = [a for a in analyses if 'error' not in a]
        
        if not successful:
            return {'error': 'Nenhuma análise bem-sucedida'}
        
        # Collect metrics
        ratings = [a['quality_assessment']['rating'] for a in successful]
        scores = [a['quality_assessment']['overall_score'] for a in successful]
        sharpness = [a['quality_assessment']['technical_metrics']['sharpness_score'] for a in successful]
        person_counts = [a['quality_assessment']['technical_metrics']['person_count'] for a in successful]
        
        # Rating distribution
        rating_dist = {}
        for rating in ['excellent', 'good', 'fair', 'poor', 'reject']:
            rating_dist[rating] = ratings.count(rating)
        
        return {
            'total_analyzed': len(successful),
            'rating_distribution': rating_dist,
            'average_quality_score': round(sum(scores) / len(scores), 3),
            'average_sharpness': round(sum(sharpness) / len(sharpness), 1),
            'average_persons_per_image': round(sum(person_counts) / len(person_counts), 1),
            'best_image': max(successful, key=lambda x: x['quality_assessment']['overall_score'])['filename'],
            'worst_image': min(successful, key=lambda x: x['quality_assessment']['overall_score'])['filename']
        }
    
    def _generate_executive_summary(self, analyses: List[Dict]) -> str:
        """
        Gera resumo executivo
        """
        summary_stats = self._generate_summary_stats(analyses)
        
        if 'error' in summary_stats:
            return "\n## ❌ ERRO NO RESUMO EXECUTIVO\nNão foi possível gerar estatísticas.\n"
        
        summary = "\n" + "="*80 + "\n"
        summary += "## 📈 RESUMO EXECUTIVO DA DEMONSTRAÇÃO\n"
        summary += "="*80 + "\n\n"
        
        summary += f"**📊 ESTATÍSTICAS GERAIS:**\n"
        summary += f"  • Total de imagens analisadas: {summary_stats['total_analyzed']}\n"
        summary += f"  • Score médio de qualidade: {summary_stats['average_quality_score']:.1f}/1.0\n"
        summary += f"  • Nitidez média: {summary_stats['average_sharpness']:.1f}\n"
        summary += f"  • Pessoas por imagem (média): {summary_stats['average_persons_per_image']:.1f}\n\n"
        
        summary += f"**🏆 DISTRIBUIÇÃO DE QUALIDADE:**\n"
        dist = summary_stats['rating_distribution']
        for rating, count in dist.items():
            emoji = {'excellent': '🌟', 'good': '✅', 'fair': '⚖️', 'poor': '⚠️', 'reject': '❌'}
            if count > 0:
                pct = (count / summary_stats['total_analyzed']) * 100
                summary += f"  • {emoji.get(rating, '❓')} {rating.capitalize()}: {count} imagens ({pct:.1f}%)\n"
        
        summary += f"\n**🥇 DESTAQUES:**\n"
        summary += f"  • Melhor imagem: {summary_stats['best_image']}\n"
        summary += f"  • Imagem com maior atenção: {summary_stats['worst_image']}\n\n"
        
        summary += "**✅ CONCLUSÃO:**\n"
        avg_score = summary_stats['average_quality_score']
        if avg_score >= 0.7:
            summary += "  🌟 Excelente qualidade geral do dataset analisado!\n"
        elif avg_score >= 0.5:
            summary += "  ✅ Boa qualidade geral com algumas oportunidades de melhoria.\n"
        elif avg_score >= 0.3:
            summary += "  ⚖️ Qualidade mista, triagem recomendada.\n"
        else:
            summary += "  ⚠️ Dataset requer curadoria significativa.\n"
        
        summary += f"\n**🚀 SISTEMA DE ANÁLISE:**\n"
        summary += f"  • 95 features automáticas extraídas por imagem\n"
        summary += f"  • Detecção de pessoas e análise de qualidade\n"
        summary += f"  • Classificação automática e recomendações\n"
        summary += f"  • Visualizações com anotações técnicas\n"
        
        return summary


def main():
    """
    Execução principal da demonstração
    """
    showcase = ProductionShowcase()
    
    print("🚀 PHOTO CULLING SYSTEM - DEMONSTRAÇÃO DE PRODUÇÃO")
    print("Análise completa com 10 imagens selecionadas")
    print("Gerando resultados JSON, visualizações e relatórios...")
    
    results = showcase.run_production_showcase(
        input_dir="data/input",
        sample_count=10
    )
    
    print(f"\n🎯 ARQUIVOS GERADOS:")
    print(f"  📄 JSON: {results['json_path']}")
    print(f"  📝 Relatório: {results['report_path']}")
    print(f"  📊 Visualizações: {results['visualizations_dir']}")


if __name__ == "__main__":
    main()
