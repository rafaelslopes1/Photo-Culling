"""
Unified Scoring System for Photo Culling System v2.5
Balances technical quality with contextual factors for sports photography
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import json

logger = logging.getLogger(__name__)

class UnifiedScoringSystem:
    """
    Unified scoring system that balances technical quality with contextual factors.
    Specifically optimized for sports photography (running events).
    """
    
    def __init__(self):
        # Component weights for final score calculation - OPTIMIZED FOR SPORTS
        self.weights = {
            'technical_quality': 0.40,    # Blur, exposure, sharpness
            'person_quality': 0.30,       # Person-specific analysis
            'composition': 0.20,          # Framing, pose, aesthetics
            'context_bonus': 0.10         # Sports context, emotion, action
        }
        
        # Critical failure thresholds - AUTO REJECTION
        self.critical_failures = {
            'face_critical_overexposure': True,
            'torso_critical_overexposure': True,
            'severe_blur': True,                    # Laplacian < 30
            'person_severely_cropped': True,        # Cropping severity = severe
            'no_person_detected': True,             # Person count = 0
            'impossible_recovery': True             # Recovery difficulty = impossible
        }
        
        # Score thresholds for classification - ADJUSTED BASED ON USER FEEDBACK
        self.rating_thresholds = {
            'excellent': 0.85,     # Top tier photos
            'good': 0.70,          # Solid photos, minor issues  
            'acceptable': 0.60,    # Usable with some problems - RAISED from 0.55
            'poor': 0.40,          # Significant issues - RAISED from 0.35  
            'rejected': 0.0        # Critical failures or unusable
        }
        
        # Context bonuses for sports photography
        self.context_bonuses = {
            'natural_pose': 0.05,           # +5% for natural pose
            'good_action': 0.08,            # +8% for good action capture
            'excellent_composition': 0.05,   # +5% for rule of thirds, etc.
            'sharp_subject': 0.05,          # +5% for person sharper than background
            'no_overexposure': 0.03,        # +3% for no overexposure issues
            'centered_person': 0.02         # +2% for well-centered person
        }
    
    def calculate_final_score(self, all_features: Dict) -> Dict:
        """
        Calculate final unified score with detailed breakdown and recommendations
        
        Args:
            all_features: Dictionary with all extracted features from image
            
        Returns:
            Dictionary with final score, rating, issues, and recommendations
        """
        try:
            # Step 1: Check for critical failures (auto-rejection)
            critical_issues = self._check_critical_failures(all_features)
            if critical_issues:
                return self._create_rejection_result(critical_issues, all_features)
            
            # Step 2: Calculate component scores
            technical_score = self._calculate_technical_score(all_features)
            person_score = self._calculate_person_score(all_features)
            composition_score = self._calculate_composition_score(all_features)
            context_bonus = self._calculate_context_bonus(all_features)
            
            # Step 3: Calculate weighted final score
            final_score = (
                technical_score * self.weights['technical_quality'] +
                person_score * self.weights['person_quality'] +
                composition_score * self.weights['composition'] +
                context_bonus * self.weights['context_bonus']
            )
            
            # Ensure score is within bounds
            final_score = max(0.0, min(1.0, final_score))
            
            # Step 4: Classify and generate recommendations
            rating = self._classify_score(final_score)
            issues = self._identify_issues(all_features, technical_score, person_score, composition_score)
            recommendations = self._generate_recommendations(rating, issues, all_features)
            
            return {
                'final_score': final_score,
                'rating': rating,
                'component_scores': {
                    'technical': technical_score,
                    'person': person_score,
                    'composition': composition_score,
                    'context_bonus': context_bonus
                },
                'issues': issues,
                'recommendations': recommendations,
                'recoverable': self._assess_recoverability(issues),
                'ranking_priority': self._calculate_ranking_priority(final_score, issues),
                'score_breakdown': self._create_score_breakdown(
                    technical_score, person_score, composition_score, context_bonus
                )
            }
            
        except Exception as e:
            logger.error(f"Erro no cálculo do score unificado: {e}")
            return self._get_default_scoring_result()
    
    def _check_critical_failures(self, features: Dict) -> List[str]:
        """Check for critical failures that warrant automatic rejection"""
        failures = []
        
        # Check overexposure critical failures
        if features.get('face_critical_overexposure', False):
            failures.append('face_critical_overexposure')
        if features.get('torso_critical_overexposure', False):
            failures.append('torso_critical_overexposure')
        
        # Check blur critical failure
        blur_score = features.get('sharpness_laplacian', 100)
        if blur_score < 30:  # Very blurry
            failures.append('severe_blur')
        
        # Check person detection
        person_count = features.get('person_count', 0)
        if person_count == 0:
            failures.append('no_person_detected')
        
        # Check cropping severity
        cropping_severity = features.get('cropping_severity', 'none')
        if cropping_severity == 'severe':
            failures.append('person_severely_cropped')
        
        # Check recovery possibility
        recovery_difficulty = features.get('recovery_difficulty', 'unknown')
        if recovery_difficulty == 'impossible':
            failures.append('impossible_recovery')
        
        return failures
    
    def _calculate_technical_score(self, features: Dict) -> float:
        """Calculate technical quality score (blur, exposure, sharpness)"""
        scores = []
        
        # Blur/Sharpness score (40% of technical)
        blur_score = features.get('sharpness_laplacian', 0)
        if blur_score >= 100:
            blur_component = 1.0
        elif blur_score >= 75:
            blur_component = 0.8
        elif blur_score >= 50:
            blur_component = 0.6
        else:
            blur_component = max(0.0, blur_score / 50.0 * 0.6)
        scores.append(('blur', blur_component, 0.4))
        
        # Exposure quality score (35% of technical)
        exposure_score = features.get('exposure_quality_score', 0.5)
        scores.append(('exposure', exposure_score, 0.35))
        
        # Person relative sharpness (25% of technical)
        person_sharpness = features.get('person_relative_sharpness', 0.5)
        scores.append(('person_sharpness', person_sharpness, 0.25))
        
        # Calculate weighted technical score
        technical_score = sum(score * weight for _, score, weight in scores)
        
        # CRITICAL OVEREXPOSURE PENALTY - especially for face
        face_critical_ratio = features.get('overexposure_face_critical_ratio', 0.0)
        torso_critical_ratio = features.get('overexposure_torso_critical_ratio', 0.0)
        
        # Heavy penalty for face overexposure (more critical than torso)
        if face_critical_ratio > 0.15:  # 15% face overexposure
            face_penalty = min(0.3, face_critical_ratio * 2.0)  # Up to 30% penalty
            technical_score *= (1.0 - face_penalty)
        
        # Moderate penalty for torso overexposure  
        if torso_critical_ratio > 0.25:  # 25% torso overexposure
            torso_penalty = min(0.2, (torso_critical_ratio - 0.25) * 1.5)  # Up to 20% penalty
            technical_score *= (1.0 - torso_penalty)
        
        return max(0.0, min(1.0, technical_score))
    
    def _calculate_person_score(self, features: Dict) -> float:
        """Calculate person-specific quality score"""
        scores = []
        
        # Person quality score (50% of person)
        person_quality = features.get('person_quality_score', 0.5)
        scores.append(('quality', person_quality, 0.5))
        
        # Cropping issues (25% of person) 
        cropping_severity = features.get('cropping_severity', 'none')
        if cropping_severity == 'none':
            cropping_score = 1.0
        elif cropping_severity == 'minor':
            cropping_score = 0.8
        elif cropping_severity == 'moderate':
            cropping_score = 0.5
        else:  # severe
            cropping_score = 0.0
        scores.append(('cropping', cropping_score, 0.25))
        
        # Pose quality (25% of person)
        pose_score = features.get('pose_naturalness_score', 0.5)
        scores.append(('pose', pose_score, 0.25))
        
        # Calculate weighted person score
        person_score = sum(score * weight for _, score, weight in scores)
        return max(0.0, min(1.0, person_score))
    
    def _calculate_composition_score(self, features: Dict) -> float:
        """Calculate composition and aesthetic score"""
        scores = []
        
        # Overall composition score (60% of composition)
        composition = features.get('composition_score', 0.5)
        scores.append(('composition', composition, 0.6))
        
        # Aesthetic score (40% of composition)
        aesthetic = features.get('aesthetic_score', 0.5)
        scores.append(('aesthetic', aesthetic, 0.4))
        
        # Calculate weighted composition score
        composition_score = sum(score * weight for _, score, weight in scores)
        return max(0.0, min(1.0, composition_score))
    
    def _calculate_context_bonus(self, features: Dict) -> float:
        """Calculate context bonuses for sports photography"""
        total_bonus = 0.0
        applied_bonuses = []
        
        # Natural pose bonus
        pose_naturalness = features.get('pose_naturalness', 'unknown')
        if pose_naturalness == 'natural':
            total_bonus += self.context_bonuses['natural_pose']
            applied_bonuses.append('natural_pose')
        
        # Good composition bonus
        composition_score = features.get('composition_score', 0.0)
        if composition_score > 0.8:
            total_bonus += self.context_bonuses['excellent_composition']
            applied_bonuses.append('excellent_composition')
        
        # Sharp subject bonus
        person_relative_sharpness = features.get('person_relative_sharpness', 0.0)
        if person_relative_sharpness > 0.8:
            total_bonus += self.context_bonuses['sharp_subject']
            applied_bonuses.append('sharp_subject')
        
        # No overexposure bonus
        overall_critical_overexposure = features.get('overall_critical_overexposure', True)
        if not overall_critical_overexposure:
            total_bonus += self.context_bonuses['no_overexposure']
            applied_bonuses.append('no_overexposure')
        
        # Centered person bonus
        person_analysis_data = features.get('person_analysis_data', {})
        if isinstance(person_analysis_data, str):
            try:
                import json
                person_analysis_data = json.loads(person_analysis_data)
            except:
                person_analysis_data = {}
        
        centrality = person_analysis_data.get('centrality', 0.0) if isinstance(person_analysis_data, dict) else 0.0
        if isinstance(centrality, (int, float)) and centrality > 0.85:
            total_bonus += self.context_bonuses['centered_person']
            applied_bonuses.append('centered_person')
        
        # Cap total bonus at base score (no bonus can exceed the base weighted score)
        return min(0.15, total_bonus)  # Max 15% bonus
    
    def _classify_score(self, score: float) -> str:
        """Classify final score into rating categories"""
        if score >= self.rating_thresholds['excellent']:
            return 'excellent'
        elif score >= self.rating_thresholds['good']:
            return 'good'
        elif score >= self.rating_thresholds['acceptable']:
            return 'acceptable'
        elif score >= self.rating_thresholds['poor']:
            return 'poor'
        else:
            return 'rejected'
    
    def _identify_issues(self, features: Dict, tech_score: float, 
                        person_score: float, comp_score: float) -> List[Dict]:
        """Identify specific issues with the image"""
        issues = []
        
        # Technical issues
        if tech_score < 0.6:
            blur_score = features.get('sharpness_laplacian', 0)
            if blur_score < 75:
                issues.append({
                    'type': 'technical',
                    'issue': 'blur',
                    'severity': 'high' if blur_score < 50 else 'medium',
                    'description': 'Imagem com desfoque significativo'
                })
            
            exposure_score = features.get('exposure_quality_score', 1.0)
            if exposure_score < 0.5:
                issues.append({
                    'type': 'technical', 
                    'issue': 'exposure',
                    'severity': 'high' if exposure_score < 0.3 else 'medium',
                    'description': 'Problemas de exposição (muito clara ou escura)'
                })
        
        # Person-specific issues
        if person_score < 0.6:
            # Overexposure issues
            if features.get('face_critical_overexposure', False):
                issues.append({
                    'type': 'person',
                    'issue': 'face_overexposure',
                    'severity': 'critical',
                    'description': 'Rosto com superexposição crítica'
                })
            
            if features.get('torso_critical_overexposure', False):
                issues.append({
                    'type': 'person',
                    'issue': 'torso_overexposure', 
                    'severity': 'critical',
                    'description': 'Torso com superexposição crítica'
                })
            
            # Cropping issues
            cropping_severity = features.get('cropping_severity', 'none')
            if cropping_severity != 'none':
                issues.append({
                    'type': 'person',
                    'issue': 'cropping',
                    'severity': cropping_severity,
                    'description': f'Pessoa cortada nas bordas ({cropping_severity})'
                })
        
        # Composition issues
        if comp_score < 0.6:
            issues.append({
                'type': 'composition',
                'issue': 'framing',
                'severity': 'medium',
                'description': 'Composição e enquadramento podem ser melhorados'
            })
        
        return issues
    
    def _generate_recommendations(self, rating: str, issues: List[Dict], 
                                features: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        if rating == 'rejected':
            recommendations.append('Rejeitar: problemas críticos irrecuperáveis')
            return recommendations
        
        # Recommendations based on issues
        for issue in issues:
            if issue['issue'] == 'blur':
                if issue['severity'] == 'high':
                    recommendations.append('Considerar rejeição: desfoque excessivo')
                else:
                    recommendations.append('Aplicar sharpening na pós-produção')
            
            elif issue['issue'] == 'face_overexposure':
                recommendations.append('Reduzir exposição do rosto na edição')
            
            elif issue['issue'] == 'torso_overexposure':
                recommendations.append('Recuperar detalhes do torso nas altas luzes')
            
            elif issue['issue'] == 'cropping':
                if issue['severity'] == 'severe':
                    recommendations.append('Considerar rejeição: corte severo')
                else:
                    recommendations.append('Aceitável: corte menor pode ser ignorado')
        
        # Positive recommendations
        if rating in ['excellent', 'good']:
            recommendations.append('Foto aprovada: qualidade adequada para uso')
        elif rating == 'acceptable':
            recommendations.append('Revisar: aceitável mas pode precisar de ajustes')
        
        return recommendations
    
    def _assess_recoverability(self, issues: List[Dict]) -> bool:
        """Assess if issues are recoverable in post-processing"""
        critical_issues = [issue for issue in issues if issue.get('severity') == 'critical']
        severe_issues = [issue for issue in issues if issue.get('severity') == 'severe']
        
        # If there are critical or severe issues, it's not easily recoverable
        return len(critical_issues) == 0 and len(severe_issues) <= 1
    
    def _calculate_ranking_priority(self, final_score: float, issues: List[Dict]) -> int:
        """Calculate ranking priority (1-100, higher is better)"""
        # Base priority from score
        base_priority = int(final_score * 80)  # 0-80 range
        
        # Bonuses for good photos
        if final_score >= 0.85:
            base_priority += 15
        elif final_score >= 0.70:
            base_priority += 10
        elif final_score >= 0.50:
            base_priority += 5
        
        # Penalties for issues
        critical_issues = len([i for i in issues if i.get('severity') == 'critical'])
        severe_issues = len([i for i in issues if i.get('severity') == 'severe'])
        
        base_priority -= (critical_issues * 20 + severe_issues * 10)
        
        return max(1, min(100, base_priority))
    
    def _create_score_breakdown(self, tech: float, person: float, 
                              comp: float, bonus: float) -> Dict:
        """Create detailed score breakdown for analysis"""
        return {
            'technical_quality': {
                'score': tech,
                'weight': self.weights['technical_quality'],
                'contribution': tech * self.weights['technical_quality']
            },
            'person_quality': {
                'score': person,
                'weight': self.weights['person_quality'],
                'contribution': person * self.weights['person_quality']
            },
            'composition': {
                'score': comp,
                'weight': self.weights['composition'],
                'contribution': comp * self.weights['composition']
            },
            'context_bonus': {
                'score': bonus,
                'weight': self.weights['context_bonus'],
                'contribution': bonus * self.weights['context_bonus']
            }
        }
    
    def _create_rejection_result(self, critical_issues: List[str], 
                               features: Dict) -> Dict:
        """Create result for rejected images"""
        main_reason = critical_issues[0] if critical_issues else 'unknown'
        
        return {
            'final_score': 0.0,
            'rating': 'rejected',
            'component_scores': {
                'technical': 0.0,
                'person': 0.0,
                'composition': 0.0,
                'context_bonus': 0.0
            },
            'issues': [
                {
                    'type': 'critical',
                    'issue': issue,
                    'severity': 'critical',
                    'description': f'Falha crítica: {issue}'
                } for issue in critical_issues
            ],
            'recommendations': ['Rejeitar: problemas críticos irrecuperáveis'],
            'recoverable': False,
            'ranking_priority': 1,
            'critical_failures': critical_issues,
            'main_rejection_reason': main_reason
        }
    
    def _get_default_scoring_result(self) -> Dict:
        """Return default result in case of scoring failure"""
        return {
            'final_score': 0.0,
            'rating': 'unknown',
            'component_scores': {
                'technical': 0.0,
                'person': 0.0,
                'composition': 0.0,
                'context_bonus': 0.0
            },
            'issues': [
                {
                    'type': 'system',
                    'issue': 'scoring_failure',
                    'severity': 'critical',
                    'description': 'Falha no sistema de scoring'
                }
            ],
            'recommendations': ['Revisão manual necessária'],
            'recoverable': False,
            'ranking_priority': 1,
            'error': 'scoring_system_failure'
        }

# Test function for system validation
def test_unified_scoring_system():
    """Test the unified scoring system with sample data"""
    scoring_system = UnifiedScoringSystem()
    
    # Test with sample features (similar to IMG_0001.JPG)
    sample_features = {
        'sharpness_laplacian': 143.44,
        'exposure_quality_score': 0.336,
        'person_quality_score': 0.545,
        'composition_score': 0.919,
        'aesthetic_score': 0.840,
        'person_relative_sharpness': 0.902,
        'cropping_severity': 'none',
        'pose_naturalness': 'natural',
        'face_critical_overexposure': True,  # This should trigger rejection
        'torso_critical_overexposure': True,
        'overall_critical_overexposure': True,
        'person_count': 1,
        'person_analysis_data': {'centrality': 0.907}
    }
    
    result = scoring_system.calculate_final_score(sample_features)
    
    print("🧪 TESTE DO UNIFIED SCORING SYSTEM")
    print("=" * 50)
    print(f"Final Score: {result['final_score']:.3f}")
    print(f"Rating: {result['rating']}")
    print(f"Recoverable: {result['recoverable']}")
    print(f"Ranking Priority: {result['ranking_priority']}")
    print("\nIssues found:")
    for issue in result['issues']:
        print(f"  - {issue['type']}: {issue['description']}")
    print("\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  - {rec}")
    
    return result

if __name__ == "__main__":
    test_unified_scoring_system()
