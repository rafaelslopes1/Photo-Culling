#!/usr/bin/env python3
"""
Photo Culling System - Análise Automática de Avaliações
Gera relatórios periódicos das avaliações manuais realizadas
"""

import sqlite3
import json
import datetime
from pathlib import Path
from typing import Dict, List, Any

class EvaluationAnalyzer:
    """
    Analyzer for manual photo evaluations
    """
    
    def __init__(self, db_path: str = "backend/expert_evaluations.db"):
        self.db_path = db_path
        
    def connect_db(self) -> sqlite3.Connection:
        """Connect to evaluation database"""
        if not Path(self.db_path).exists():
            raise Exception(f"Banco de dados não encontrado: {self.db_path}")
        return sqlite3.connect(self.db_path)
        
    def get_all_evaluations(self) -> List[Dict]:
        """Get all evaluations from database"""
        conn = self.connect_db()
        conn.row_factory = sqlite3.Row
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM expert_evaluation 
            ORDER BY timestamp DESC
        """)
        
        evaluations = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return evaluations
        
    def calculate_basic_stats(self, evaluations: List[Dict]) -> Dict:
        """Calculate basic statistics"""
        if not evaluations:
            return {"error": "Nenhuma avaliação encontrada"}
            
        total = len(evaluations)
        quality_scores = [eval.get('overall_quality', 0) for eval in evaluations]
        confidence_scores = [eval.get('confidence_level', 0) for eval in evaluations]
        
        # Quality distribution
        quality_dist = {i: quality_scores.count(i) for i in range(1, 6)}
        
        # Rejection analysis
        rejected = sum(1 for eval in evaluations if not eval.get('approve_for_portfolio', False))
        rejection_rate = (rejected / total) * 100 if total > 0 else 0
        
        return {
            "total_evaluations": total,
            "quality_average": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "confidence_average": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
            "quality_distribution": quality_dist,
            "rejection_rate": rejection_rate,
            "rejected_count": rejected
        }
        
    def analyze_by_people_count(self, evaluations: List[Dict]) -> Dict:
        """Analyze quality by number of people in photos"""
        people_analysis = {}
        
        for eval in evaluations:
            people_count = eval.get('people_count', 'unknown')
            quality = eval.get('overall_quality', 0)
            
            if people_count not in people_analysis:
                people_analysis[people_count] = {
                    'count': 0,
                    'quality_scores': [],
                    'quality_average': 0
                }
            
            people_analysis[people_count]['count'] += 1
            people_analysis[people_count]['quality_scores'].append(quality)
        
        # Calculate averages
        for category in people_analysis:
            scores = people_analysis[category]['quality_scores']
            people_analysis[category]['quality_average'] = sum(scores) / len(scores) if scores else 0
            
        return people_analysis
        
    def analyze_by_context(self, evaluations: List[Dict]) -> Dict:
        """Analyze quality by photo context/lighting"""
        context_analysis = {}
        
        for eval in evaluations:
            context = eval.get('photo_context', 'unknown')
            quality = eval.get('overall_quality', 0)
            
            if context not in context_analysis:
                context_analysis[context] = {
                    'count': 0,
                    'quality_scores': [],
                    'quality_average': 0
                }
            
            context_analysis[context]['count'] += 1
            context_analysis[context]['quality_scores'].append(quality)
        
        # Calculate averages
        for category in context_analysis:
            scores = context_analysis[category]['quality_scores']
            context_analysis[category]['quality_average'] = sum(scores) / len(scores) if scores else 0
            
        return context_analysis
        
    def generate_insights(self, stats: Dict, people_analysis: Dict, context_analysis: Dict) -> List[str]:
        """Generate actionable insights from analysis"""
        insights = []
        
        # Sample size analysis
        total = stats.get('total_evaluations', 0)
        if total < 25:
            insights.append(f"⚠️  Amostra pequena ({total} avaliações). Necessário mínimo 25 para análises confiáveis.")
        elif total < 50:
            insights.append(f"📊 Amostra moderada ({total} avaliações). Próximo milestone: 50 avaliações.")
        else:
            insights.append(f"✅ Amostra robusta ({total} avaliações). Análises estatisticamente válidas.")
            
        # Quality analysis
        avg_quality = stats.get('quality_average', 0)
        if avg_quality < 2.5:
            insights.append("📉 Qualidade média baixa. Revisar critérios de pré-seleção.")
        elif avg_quality > 4.0:
            insights.append("📈 Qualidade média alta. Padrões excelentes mantidos.")
        else:
            insights.append("⚖️  Qualidade média equilibrada. Distribuição normal esperada.")
            
        # Rejection rate analysis
        rejection_rate = stats.get('rejection_rate', 0)
        if rejection_rate > 70:
            insights.append(f"🚨 Taxa de rejeição alta ({rejection_rate:.1f}%). Implementar filtros automáticos.")
        elif rejection_rate < 30:
            insights.append(f"✅ Taxa de rejeição baixa ({rejection_rate:.1f}%). Boa qualidade de entrada.")
        else:
            insights.append(f"⚖️  Taxa de rejeição normal ({rejection_rate:.1f}%).")
            
        # Context analysis
        best_context = max(context_analysis.items(), key=lambda x: x[1]['quality_average']) if context_analysis else None
        if best_context:
            context_name, context_data = best_context
            insights.append(f"🏆 Melhor contexto: {context_name} (qualidade média: {context_data['quality_average']:.2f})")
            
        return insights
        
    def generate_report(self) -> Dict:
        """Generate complete analysis report"""
        try:
            evaluations = self.get_all_evaluations()
            
            if not evaluations:
                return {
                    "status": "error",
                    "message": "Nenhuma avaliação encontrada no banco de dados"
                }
                
            stats = self.calculate_basic_stats(evaluations)
            people_analysis = self.analyze_by_people_count(evaluations)
            context_analysis = self.analyze_by_context(evaluations)
            insights = self.generate_insights(stats, people_analysis, context_analysis)
            
            report = {
                "status": "success",
                "generated_at": datetime.datetime.now().isoformat(),
                "basic_stats": stats,
                "people_analysis": people_analysis,
                "context_analysis": context_analysis,
                "insights": insights,
                "recommendations": self._generate_recommendations(stats, insights)
            }
            
            return report
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Erro ao gerar relatório: {str(e)}"
            }
            
    def _generate_recommendations(self, stats: Dict, insights: List[str]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        total = stats.get('total_evaluations', 0)
        
        if total < 25:
            recommendations.append("📋 Prioridade 1: Acelerar coleta de avaliações (meta: 5-10 por dia)")
            recommendations.append("🎯 Estabelecer rotina diária de avaliação")
            
        if stats.get('rejection_rate', 0) > 60:
            recommendations.append("🔧 Implementar filtros automáticos básicos (blur, exposição)")
            recommendations.append("📚 Revisar guidelines de fotografia")
            
        if total >= 25:
            recommendations.append("🤖 Considerar treinamento de modelo básico de IA")
            recommendations.append("📊 Implementar análises de correlação entre critérios")
            
        if total >= 50:
            recommendations.append("🎯 Iniciar benchmarking de qualidade por categoria")
            recommendations.append("📈 Análise temporal de melhorias")
            
        return recommendations
        
    def save_report(self, filename: str = None) -> str:
        """Save analysis report to file"""
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            filename = f"evaluation_analysis_{timestamp}.json"
            
        report = self.generate_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        return filename

def main():
    """Main execution function"""
    print("🔍 Analisando avaliações manuais...")
    
    analyzer = EvaluationAnalyzer()
    
    try:
        report = analyzer.generate_report()
        
        if report["status"] == "error":
            print(f"❌ Erro: {report['message']}")
            return
            
        # Print summary
        stats = report["basic_stats"]
        print(f"\n📊 RESUMO EXECUTIVO")
        print(f"═══════════════════")
        print(f"Total de avaliações: {stats['total_evaluations']}")
        print(f"Qualidade média: {stats['quality_average']:.2f}/5")
        print(f"Confiança média: {stats['confidence_average']:.1f}%")
        print(f"Taxa de rejeição: {stats['rejection_rate']:.1f}%")
        
        print(f"\n💡 INSIGHTS PRINCIPAIS")
        print(f"═══════════════════")
        for insight in report["insights"]:
            print(f"• {insight}")
            
        print(f"\n🎯 RECOMENDAÇÕES")
        print(f"═══════════════")
        for rec in report["recommendations"]:
            print(f"• {rec}")
            
        # Save detailed report
        filename = analyzer.save_report()
        print(f"\n💾 Relatório detalhado salvo: {filename}")
        
    except Exception as e:
        print(f"❌ Erro na análise: {e}")

if __name__ == "__main__":
    main()
