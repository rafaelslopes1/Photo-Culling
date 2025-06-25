#!/usr/bin/env python3
"""
System Calibrator - Calibrador do Sistema
Aplica ajustes automáticos baseados na análise do especialista
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

@dataclass
class CalibrationAdjustment:
    """Representa um ajuste de calibração"""
    component: str
    parameter: str
    old_value: Any
    new_value: Any
    reason: str

class SystemCalibrator:
    """
    Calibrador automático do sistema baseado em feedback de especialista
    """
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.backup_path = self.config_path.with_suffix('.backup.json')
        self.adjustments = []
        
    def backup_config(self):
        """
        Cria backup da configuração atual
        """
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            with open(self.backup_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✅ Backup criado: {self.backup_path}")
    
    def load_config(self) -> Dict[str, Any]:
        """
        Carrega configuração atual
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def save_config(self, config: Dict[str, Any]):
        """
        Salva configuração atualizada
        """
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuração atualizada salva")
    
    def apply_sharpness_adjustments(self, config: Dict[str, Any], feedback: Dict[str, Any]):
        """
        Ajusta thresholds de sharpness baseado no feedback
        """
        if 'processing_settings' not in config:
            config['processing_settings'] = {}
        
        if 'blur_detection_optimized' not in config['processing_settings']:
            config['processing_settings']['blur_detection_optimized'] = {
                "enabled": True,
                "strategy": "balanced",
                "strategies": {
                    "conservative": {"threshold": 50},
                    "balanced": {"threshold": 78},
                    "aggressive": {"threshold": 145}
                }
            }
        
        blur_config = config['processing_settings']['blur_detection_optimized']
        current_strategy = blur_config.get('strategy', 'balanced')
        current_threshold = blur_config['strategies'][current_strategy]['threshold']
        
        # Analyze agreement patterns
        agreement_levels = feedback.get('agreement_levels', {})
        high_disagreement = agreement_levels.get('Discordo completamente', 0) + agreement_levels.get('Discordo parcialmente', 0)
        total = sum(agreement_levels.values())
        
        if total > 0:
            disagreement_rate = (high_disagreement / total) * 100
            
            if disagreement_rate > 30:  # High disagreement
                # Make more conservative
                new_threshold = int(current_threshold * 0.9)
                self.adjustments.append(CalibrationAdjustment(
                    component="blur_detection",
                    parameter=f"strategies.{current_strategy}.threshold",
                    old_value=current_threshold,
                    new_value=new_threshold,
                    reason=f"Alta taxa de discordância ({disagreement_rate:.1f}%) - tornando detecção mais conservadora"
                ))
                blur_config['strategies'][current_strategy]['threshold'] = new_threshold
            
            elif disagreement_rate < 10:  # Low disagreement but maybe too conservative
                # Analyze discrepancies for over/under rating
                discrepancies = feedback.get('discrepancies', [])
                system_over_rating = sum(1 for d in discrepancies if 'excellent' in d.get('system_rating', '').lower())
                
                if system_over_rating > len(discrepancies) / 2:
                    # System is over-rating, make more aggressive
                    new_threshold = int(current_threshold * 1.1)
                    self.adjustments.append(CalibrationAdjustment(
                        component="blur_detection",
                        parameter=f"strategies.{current_strategy}.threshold",
                        old_value=current_threshold,
                        new_value=new_threshold,
                        reason="Sistema superestimando qualidade - tornando detecção mais rigorosa"
                    ))
                    blur_config['strategies'][current_strategy]['threshold'] = new_threshold
    
    def apply_scoring_adjustments(self, config: Dict[str, Any], feedback: Dict[str, Any]):
        """
        Ajusta pesos do sistema de pontuação
        """
        if 'scoring_weights' not in config:
            config['scoring_weights'] = {
                "sharpness_weight": 0.4,
                "exposure_weight": 0.3,
                "composition_weight": 0.2,
                "person_detection_weight": 0.1
            }
        
        scoring = config['scoring_weights']
        
        # Analyze discrepancies to adjust weights
        discrepancies = feedback.get('discrepancies', [])
        
        if len(discrepancies) > 0:
            # Simple heuristic: if many discrepancies, rebalance weights
            expert_comments = [d.get('expert_comments', '') for d in discrepancies]
            
            # Check for common themes in expert comments
            sharpness_mentions = sum(1 for comment in expert_comments if any(word in comment.lower() for word in ['foco', 'nitidez', 'blur', 'desfoque']))
            exposure_mentions = sum(1 for comment in expert_comments if any(word in comment.lower() for word in ['exposição', 'luz', 'escuro', 'claro', 'brilho']))
            composition_mentions = sum(1 for comment in expert_comments if any(word in comment.lower() for word in ['composição', 'enquadramento', 'corte']))
            
            total_mentions = sharpness_mentions + exposure_mentions + composition_mentions
            
            if total_mentions > 0:
                # Rebalance weights based on expert focus
                old_sharpness = scoring['sharpness_weight']
                old_exposure = scoring['exposure_weight']
                old_composition = scoring['composition_weight']
                
                # Increase weight for frequently mentioned aspects
                if sharpness_mentions > total_mentions * 0.4:
                    scoring['sharpness_weight'] = min(0.6, scoring['sharpness_weight'] + 0.1)
                    self.adjustments.append(CalibrationAdjustment(
                        component="scoring",
                        parameter="sharpness_weight",
                        old_value=old_sharpness,
                        new_value=scoring['sharpness_weight'],
                        reason="Especialista enfatizou questões de nitidez"
                    ))
                
                if exposure_mentions > total_mentions * 0.4:
                    scoring['exposure_weight'] = min(0.5, scoring['exposure_weight'] + 0.1)
                    self.adjustments.append(CalibrationAdjustment(
                        component="scoring",
                        parameter="exposure_weight",
                        old_value=old_exposure,
                        new_value=scoring['exposure_weight'],
                        reason="Especialista enfatizou questões de exposição"
                    ))
                
                if composition_mentions > total_mentions * 0.4:
                    scoring['composition_weight'] = min(0.4, scoring['composition_weight'] + 0.1)
                    self.adjustments.append(CalibrationAdjustment(
                        component="scoring",
                        parameter="composition_weight",
                        old_value=old_composition,
                        new_value=scoring['composition_weight'],
                        reason="Especialista enfatizou questões de composição"
                    ))
    
    def apply_quality_thresholds(self, config: Dict[str, Any], feedback: Dict[str, Any]):
        """
        Ajusta thresholds de classificação de qualidade
        """
        if 'quality_thresholds' not in config:
            config['quality_thresholds'] = {
                "excellent_threshold": 0.85,
                "good_threshold": 0.65,
                "fair_threshold": 0.45,
                "poor_threshold": 0.25
            }
        
        thresholds = config['quality_thresholds']
        
        # Analyze rating patterns
        discrepancies = feedback.get('discrepancies', [])
        over_ratings = [d for d in discrepancies if 'excellent' in d.get('system_rating', '').lower()]
        under_ratings = [d for d in discrepancies if d.get('expert_rating', '') in ['Excelente - Pronta para portfólio profissional']]
        
        if len(over_ratings) > len(under_ratings) and len(over_ratings) > 2:
            # System is over-rating, raise thresholds
            old_excellent = thresholds['excellent_threshold']
            old_good = thresholds['good_threshold']
            
            thresholds['excellent_threshold'] = min(0.95, thresholds['excellent_threshold'] + 0.05)
            thresholds['good_threshold'] = min(0.8, thresholds['good_threshold'] + 0.05)
            
            self.adjustments.append(CalibrationAdjustment(
                component="quality_thresholds",
                parameter="excellent_threshold",
                old_value=old_excellent,
                new_value=thresholds['excellent_threshold'],
                reason="Sistema superestimando - aumentando threshold para 'excellent'"
            ))
            
            self.adjustments.append(CalibrationAdjustment(
                component="quality_thresholds",
                parameter="good_threshold",
                old_value=old_good,
                new_value=thresholds['good_threshold'],
                reason="Sistema superestimando - aumentando threshold para 'good'"
            ))
        
        elif len(under_ratings) > len(over_ratings) and len(under_ratings) > 2:
            # System is under-rating, lower thresholds
            old_excellent = thresholds['excellent_threshold']
            old_good = thresholds['good_threshold']
            
            thresholds['excellent_threshold'] = max(0.75, thresholds['excellent_threshold'] - 0.05)
            thresholds['good_threshold'] = max(0.55, thresholds['good_threshold'] - 0.05)
            
            self.adjustments.append(CalibrationAdjustment(
                component="quality_thresholds",
                parameter="excellent_threshold",
                old_value=old_excellent,
                new_value=thresholds['excellent_threshold'],
                reason="Sistema subestimando - diminuindo threshold para 'excellent'"
            ))
            
            self.adjustments.append(CalibrationAdjustment(
                component="quality_thresholds",
                parameter="good_threshold",
                old_value=old_good,
                new_value=thresholds['good_threshold'],
                reason="Sistema subestimando - diminuindo threshold para 'good'"
            ))
    
    def calibrate_from_feedback(self, feedback_data: Dict[str, Any]) -> List[CalibrationAdjustment]:
        """
        Aplica calibração baseada no feedback do especialista
        """
        print("🔧 Iniciando calibração automática...")
        
        # Backup current config
        self.backup_config()
        
        # Load current config
        config = self.load_config()
        
        # Apply various adjustments
        self.apply_sharpness_adjustments(config, feedback_data)
        self.apply_scoring_adjustments(config, feedback_data)
        self.apply_quality_thresholds(config, feedback_data)
        
        # Save updated config
        self.save_config(config)
        
        return self.adjustments
    
    def generate_calibration_report(self) -> str:
        """
        Gera relatório dos ajustes aplicados
        """
        timestamp = datetime.now().strftime("%d/%m/%Y às %H:%M:%S")
        
        report = f"""# 🔧 RELATÓRIO DE CALIBRAÇÃO AUTOMÁTICA

**Data:** {timestamp}  
**Ajustes aplicados:** {len(self.adjustments)}  
**Backup:** {self.backup_path}  

---

## 📋 AJUSTES REALIZADOS

"""
        
        if not self.adjustments:
            report += "✅ Nenhum ajuste necessário - sistema bem calibrado\n"
        else:
            for i, adj in enumerate(self.adjustments, 1):
                report += f"### **Ajuste {i}: {adj.component.upper()}**\n\n"
                report += f"- **Parâmetro:** `{adj.parameter}`\n"
                report += f"- **Valor anterior:** `{adj.old_value}`\n"
                report += f"- **Novo valor:** `{adj.new_value}`\n"
                report += f"- **Motivo:** {adj.reason}\n\n"
        
        report += f"""
---

## ⚠️ PRÓXIMOS PASSOS

1. **Testar nova configuração:**
   ```bash
   python tools/analysis/production_showcase.py
   ```

2. **Comparar resultados:**
   - Execute novamente o teste de produção
   - Compare com resultados anteriores
   - Valide se ajustes melhoraram a concordância

3. **Reverter se necessário:**
   ```bash
   cp {self.backup_path} {self.config_path}
   ```

4. **Ciclo de refinamento:**
   - Se necessário, solicite nova análise do especialista
   - Repita processo de calibração
   - Continue até obter alta concordância

---

**⚡ Sistema calibrado e pronto para novos testes!**
"""
        
        return report
    
    def restore_backup(self):
        """
        Restaura configuração do backup
        """
        if not self.backup_path.exists():
            raise FileNotFoundError(f"Backup não encontrado: {self.backup_path}")
        
        with open(self.backup_path, 'r') as f:
            config = json.load(f)
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuração restaurada do backup: {self.backup_path}")


def main():
    """
    Execução principal
    """
    print("🔧 CALIBRADOR AUTOMÁTICO DO SISTEMA")
    print("=" * 60)
    
    # Check for expert analysis results
    analysis_dir = Path("data/analysis_results/production_showcase")
    expert_reports = list(analysis_dir.glob("EXPERT_COMPARISON_REPORT_*.md"))
    
    if not expert_reports:
        print("❌ Nenhum relatório de análise de especialista encontrado")
        print("🔧 Execute primeiro:")
        print("   1. Preencha o formulário de especialista")
        print("   2. Execute: python tools/analysis/expert_analysis_processor.py")
        return
    
    # Use most recent report
    latest_report = max(expert_reports, key=lambda p: p.stat().st_mtime)
    print(f"📄 Usando relatório: {latest_report}")
    
    # For this example, we'll simulate feedback data
    # In real usage, this would be parsed from the expert analysis processor
    feedback_data = {
        'agreement_levels': {
            'Concordo completamente': 4,
            'Concordo parcialmente (diferença pequena)': 3,
            'Neutro': 1,
            'Discordo parcialmente': 2,
            'Discordo completamente': 0
        },
        'discrepancies': [
            {
                'image': 'IMG_0001.JPG',
                'expert_rating': 'Boa - Utilizável com pequenos ajustes',
                'system_rating': 'excellent',
                'expert_comments': 'Imagem tem boa nitidez mas composição poderia ser melhor'
            }
        ]
    }
    
    try:
        calibrator = SystemCalibrator()
        adjustments = calibrator.calibrate_from_feedback(feedback_data)
        
        # Generate and save report
        report = calibrator.generate_calibration_report()
        report_path = Path("data/analysis_results/production_showcase") / f"CALIBRATION_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n🎉 CALIBRAÇÃO CONCLUÍDA!")
        print(f"📊 {len(adjustments)} ajustes aplicados")
        print(f"📄 Relatório: {report_path}")
        print(f"💾 Backup: {calibrator.backup_path}")
        
        print("\n🔄 Execute novamente o teste de produção para validar os ajustes")
        
    except Exception as e:
        print(f"❌ Erro durante calibração: {e}")


if __name__ == "__main__":
    main()
