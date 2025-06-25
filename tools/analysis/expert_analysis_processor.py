#!/usr/bin/env python3
"""
Expert Analysis Processor - Processador de Análise de Especialista
Ferramenta para processar e comparar a análise manual do fotógrafo com o sistema automático
"""

import sys
import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

@dataclass
class ExpertRating:
    """Avaliação de uma imagem pelo especialista"""
    image_name: str
    sharpness: str
    exposure: str
    contrast: str
    composition: str
    person_detection: str
    person_positioning: str
    overall_classification: str
    agreement_with_system: str
    comments: str

@dataclass
class SystemRating:
    """Avaliação de uma imagem pelo sistema"""
    image_name: str
    rating: str
    score: float
    sharpness: float
    exposure: str
    persons_detected: int
    dominant_score: float

class ExpertAnalysisProcessor:
    """
    Processador da análise manual do especialista
    """
    
    def __init__(self, form_path: str, results_path: str):
        self.form_path = Path(form_path)
        self.results_path = Path(results_path)
        self.expert_ratings = []
        self.system_ratings = []
        self.comparison_results = {}
        
    def parse_expert_form(self) -> List[ExpertRating]:
        """
        Parse o formulário preenchido pelo especialista
        """
        if not self.form_path.exists():
            raise FileNotFoundError(f"Formulário não encontrado: {self.form_path}")
            
        with open(self.form_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract image sections
        image_sections = re.findall(r'## \*\*IMAGEM \d+: (.+?)\*\*.*?### \*\*C\. AVALIAÇÃO PROFISSIONAL\*\*(.*?)(?=## \*\*IMAGEM|\# 📈)', content, re.DOTALL)
        
        expert_ratings = []
        
        for image_name, section in image_sections:
            # Parse responses
            rating = self._extract_checkbox_response(section, r'C1\. Classificação geral')
            agreement = self._extract_checkbox_response(section, r'C2\. Concordância com avaliação')
            comments = self._extract_text_response(section, r'C3\. Comentários específicos')
            
            # Extract technical ratings from previous sections
            full_section = re.search(rf'## \*\*IMAGEM \d+: {re.escape(image_name)}\*\*(.*?)(?=## \*\*IMAGEM|\# 📈)', content, re.DOTALL)
            if full_section:
                full_text = full_section.group(1)
                sharpness = self._extract_checkbox_response(full_text, r'A1\. Avaliação geral de nitidez')
                exposure = self._extract_checkbox_response(full_text, r'A2\. Qualidade da exposição')
                contrast = self._extract_checkbox_response(full_text, r'A3\. Avaliação do contraste')
                composition = self._extract_checkbox_response(full_text, r'B1\. Enquadramento e composição')
                person_detection = self._extract_checkbox_response(full_text, r'B2\. Qualidade da detecção')
                person_positioning = self._extract_checkbox_response(full_text, r'B3\. Avaliação do posicionamento')
            else:
                sharpness = exposure = contrast = composition = person_detection = person_positioning = "N/A"
            
            expert_rating = ExpertRating(
                image_name=image_name.strip(),
                sharpness=sharpness,
                exposure=exposure,
                contrast=contrast,
                composition=composition,
                person_detection=person_detection,
                person_positioning=person_positioning,
                overall_classification=rating,
                agreement_with_system=agreement,
                comments=comments.strip()
            )
            
            expert_ratings.append(expert_rating)
        
        return expert_ratings
    
    def _extract_checkbox_response(self, text: str, question_pattern: str) -> str:
        """
        Extrai resposta de checkbox marcada
        """
        # Find the question section
        pattern = rf'{question_pattern}:(.*?)(?=\*\*[A-Z]\d+\.|\*\*[C]\d+\.|\### |\Z)'
        match = re.search(pattern, text, re.DOTALL)
        
        if not match:
            return "N/A"
        
        section = match.group(1)
        
        # Find marked checkbox (with [x] or [X])
        marked = re.findall(r'- \[x\] (.+?)(?=\n|$)', section, re.IGNORECASE)
        
        if marked:
            return marked[0].strip()
        
        return "Não respondido"
    
    def _extract_text_response(self, text: str, question_pattern: str) -> str:
        """
        Extrai resposta de texto livre
        """
        pattern = rf'{question_pattern}:(.*?)```(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            response = match.group(2).strip()
            # Remove placeholder lines
            response = re.sub(r'^_+$', '', response, flags=re.MULTILINE)
            return response.strip()
        
        return ""
    
    def load_system_results(self) -> List[SystemRating]:
        """
        Carrega resultados do sistema automático
        """
        if not self.results_path.exists():
            raise FileNotFoundError(f"Resultados do sistema não encontrados: {self.results_path}")
        
        with open(self.results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        system_ratings = []
        
        for analysis in data.get('analyses', []):
            system_rating = SystemRating(
                image_name=analysis['filename'],
                rating=analysis['quality_assessment']['rating'],
                score=analysis['quality_assessment']['overall_score'],
                sharpness=analysis['quality_assessment']['technical_metrics']['sharpness_score'],
                exposure=analysis['quality_assessment']['technical_metrics']['exposure_level'],
                persons_detected=analysis['quality_assessment']['technical_metrics']['person_count'],
                dominant_score=analysis['quality_assessment']['technical_metrics']['dominant_person_score']
            )
            
            system_ratings.append(system_rating)
        
        return system_ratings
    
    def compare_ratings(self) -> Dict[str, Any]:
        """
        Compara avaliações do especialista com o sistema
        """
        comparison = {
            'total_images': len(self.expert_ratings),
            'agreement_levels': defaultdict(int),
            'discrepancies': [],
            'technical_accuracy': {},
            'recommendations': []
        }
        
        # Map ratings to numerical values for comparison
        rating_map = {
            'Excelente - Pronta para portfólio profissional': 5,
            'Muito boa - Adequada para uso comercial': 4,
            'Boa - Utilizável com pequenos ajustes': 3,
            'Regular - Requer edição significativa': 2,
            'Ruim - Considerar descarte': 1,
            'Péssima - Descartar': 0,
            'excellent': 5,
            'good': 3,
            'fair': 2,
            'poor': 1,
            'reject': 0
        }
        
        for expert, system in zip(self.expert_ratings, self.system_ratings):
            if expert.image_name.replace(' ', '') != system.image_name.replace(' ', ''):
                continue  # Skip if names don't match
            
            # Agreement analysis
            agreement = expert.agreement_with_system
            comparison['agreement_levels'][agreement] += 1
            
            # Rating comparison
            expert_score = rating_map.get(expert.overall_classification, -1)
            system_score = rating_map.get(system.rating, -1)
            
            if expert_score != -1 and system_score != -1:
                difference = abs(expert_score - system_score)
                
                if difference >= 2:  # Significant discrepancy
                    comparison['discrepancies'].append({
                        'image': expert.image_name,
                        'expert_rating': expert.overall_classification,
                        'system_rating': system.rating,
                        'difference': difference,
                        'expert_comments': expert.comments
                    })
        
        return comparison
    
    def generate_calibration_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """
        Gera recomendações para calibração do sistema
        """
        recommendations = []
        
        # Agreement analysis
        total = comparison['total_images']
        agreements = comparison['agreement_levels']
        
        high_agreement = agreements.get('Concordo completamente', 0) + agreements.get('Concordo parcialmente (diferença pequena)', 0)
        agreement_rate = (high_agreement / total) * 100 if total > 0 else 0
        
        if agreement_rate < 70:
            recommendations.append(f"⚠️ Taxa de concordância baixa ({agreement_rate:.1f}%) - Sistema precisa de calibração significativa")
        elif agreement_rate < 85:
            recommendations.append(f"🔧 Taxa de concordância moderada ({agreement_rate:.1f}%) - Ajustes finos necessários")
        else:
            recommendations.append(f"✅ Alta taxa de concordância ({agreement_rate:.1f}%) - Sistema bem calibrado")
        
        # Discrepancy analysis
        discrepancies = comparison['discrepancies']
        if discrepancies:
            recommendations.append(f"📊 {len(discrepancies)} discrepâncias significativas identificadas")
            
            # Analyze patterns in discrepancies
            over_ratings = [d for d in discrepancies if 'excellent' in d['system_rating'].lower()]
            under_ratings = [d for d in discrepancies if d['expert_rating'] in ['Excelente - Pronta para portfólio profissional', 'Muito boa - Adequada para uso comercial']]
            
            if len(over_ratings) > len(under_ratings):
                recommendations.append("🔽 Sistema tende a superavaliar imagens - reduzir thresholds de qualidade")
            elif len(under_ratings) > len(over_ratings):
                recommendations.append("🔼 Sistema tende a subavaliar imagens - aumentar sensibilidade para qualidade")
        
        return recommendations
    
    def generate_analysis_report(self) -> str:
        """
        Gera relatório completo da análise comparativa
        """
        timestamp = datetime.now().strftime("%d/%m/%Y às %H:%M:%S")
        
        report = f"""# 📊 RELATÓRIO DE ANÁLISE COMPARATIVA - ESPECIALISTA vs SISTEMA

**Data:** {timestamp}  
**Imagens analisadas:** {len(self.expert_ratings)}  
**Analista:** Fotógrafo Especialista  

---

## 📈 RESULTADOS DA COMPARAÇÃO

### **Taxa de Concordância Geral**
"""
        
        comparison = self.compare_ratings()
        
        total = comparison['total_images']
        for agreement, count in comparison['agreement_levels'].items():
            percentage = (count / total) * 100 if total > 0 else 0
            report += f"- **{agreement}**: {count} imagens ({percentage:.1f}%)\n"
        
        report += "\n### **Discrepâncias Significativas**\n"
        
        if comparison['discrepancies']:
            for disc in comparison['discrepancies']:
                report += f"\n**{disc['image']}:**\n"
                report += f"- Especialista: {disc['expert_rating']}\n"
                report += f"- Sistema: {disc['system_rating'].upper()}\n"
                if disc['expert_comments']:
                    report += f"- Comentários: {disc['expert_comments']}\n"
        else:
            report += "✅ Nenhuma discrepância significativa encontrada\n"
        
        # Recommendations
        recommendations = self.generate_calibration_recommendations(comparison)
        
        report += "\n## 🎯 RECOMENDAÇÕES DE CALIBRAÇÃO\n\n"
        for rec in recommendations:
            report += f"{rec}\n\n"
        
        # Detailed analysis per image
        report += "\n## 📸 ANÁLISE DETALHADA POR IMAGEM\n\n"
        
        for expert, system in zip(self.expert_ratings, self.system_ratings):
            if expert.image_name.replace(' ', '') != system.image_name.replace(' ', ''):
                continue
                
            report += f"### **{expert.image_name}**\n"
            report += f"- **Sistema**: {system.rating.upper()} ({system.score:.1f}/1.0)\n"
            report += f"- **Especialista**: {expert.overall_classification}\n"
            report += f"- **Concordância**: {expert.agreement_with_system}\n"
            
            if expert.comments:
                report += f"- **Observações**: {expert.comments}\n"
            
            report += "\n"
        
        return report
    
    def suggest_algorithm_adjustments(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sugere ajustes específicos nos algoritmos
        """
        adjustments = {
            'sharpness_thresholds': {},
            'exposure_weights': {},
            'person_detection': {},
            'scoring_weights': {},
            'new_features': []
        }
        
        # Analyze discrepancies to suggest specific adjustments
        for disc in comparison['discrepancies']:
            # Implement specific adjustment logic based on expert feedback
            pass
        
        return adjustments
    
    def process_analysis(self) -> Dict[str, Any]:
        """
        Processa análise completa
        """
        print("🔍 Processando análise do especialista...")
        
        # Parse expert form
        self.expert_ratings = self.parse_expert_form()
        print(f"✅ {len(self.expert_ratings)} avaliações do especialista processadas")
        
        # Load system results
        self.system_ratings = self.load_system_results()
        print(f"✅ {len(self.system_ratings)} avaliações do sistema carregadas")
        
        # Compare ratings
        comparison = self.compare_ratings()
        print("✅ Comparação concluída")
        
        # Generate report
        report = self.generate_analysis_report()
        
        # Save results
        output_dir = self.form_path.parent
        report_path = output_dir / f"EXPERT_COMPARISON_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Relatório salvo: {report_path}")
        
        return {
            'comparison': comparison,
            'report_path': report_path,
            'adjustments': self.suggest_algorithm_adjustments(comparison)
        }


def main():
    """
    Execução principal
    """
    print("📊 PROCESSADOR DE ANÁLISE DE ESPECIALISTA")
    print("=" * 60)
    
    # Default paths
    form_path = "data/analysis_results/production_showcase/PHOTOGRAPHER_EXPERT_ANALYSIS_FORM.md"
    results_path = "data/analysis_results/production_showcase/production_showcase_results_20250625_152323.json"
    
    # Check if form is filled
    if not Path(form_path).exists():
        print(f"❌ Formulário não encontrado: {form_path}")
        return
    
    # Check if form appears to be filled (simple heuristic)
    with open(form_path, 'r') as f:
        content = f.read()
    
    if '[x]' not in content.lower() and '[X]' not in content:
        print("⚠️ Formulário não parece estar preenchido (nenhum checkbox marcado)")
        print("🔧 Para usar esta ferramenta:")
        print("   1. Preencha o formulário marcando as opções [x]")
        print("   2. Execute novamente esta ferramenta")
        return
    
    try:
        processor = ExpertAnalysisProcessor(form_path, results_path)
        results = processor.process_analysis()
        
        print("\n🎉 ANÁLISE CONCLUÍDA!")
        print(f"📄 Relatório: {results['report_path']}")
        print("🔧 Use os resultados para calibrar o sistema")
        
    except Exception as e:
        print(f"❌ Erro durante processamento: {e}")


if __name__ == "__main__":
    main()
