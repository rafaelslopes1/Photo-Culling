#!/usr/bin/env python3
"""
Blur Detection Configuration - Consolidated
Configuração consolidada de detecção de blur baseada em análise supervisionada
"""

from typing import Dict, Union, Optional

# =============================================================================
# PRACTICAL THRESHOLDS - Recomendados para uso real
# =============================================================================

# Estratégias práticas baseadas em análise híbrida
CONSERVATIVE_PRACTICAL = 30   # Remove apenas blur extremo (10-15% removal)
MODERATE_PRACTICAL = 60       # Threshold moderado, boa separação (25-35% removal) - RECOMENDADO
QUALITY_FOCUSED = 100         # Foco em qualidade técnica (40-60% removal)

# Threshold padrão recomendado
DEFAULT_PRACTICAL_THRESHOLD = CONSERVATIVE_PRACTICAL

# =============================================================================
# GENERIC THRESHOLDS - Baseados em análise estatística geral
# =============================================================================

CONSERVATIVE_THRESHOLD = 50   # Conservador - apenas casos muito borrados
BALANCED_THRESHOLD = 78       # Balanceado - uso geral
AGGRESSIVE_THRESHOLD = 145    # Agressivo - alta qualidade
VERY_AGGRESSIVE_THRESHOLD = 98

# =============================================================================
# CUSTOM THRESHOLDS - Baseados nos dados específicos do usuário
# =============================================================================

CUSTOM_CONSERVATIVE_THRESHOLD = 151  # Captura 50% das rejeições por blur
CUSTOM_BALANCED_THRESHOLD = 218      # Captura 75% das rejeições por blur
CUSTOM_AGGRESSIVE_THRESHOLD = 387    # Captura 90% das rejeições por blur

# =============================================================================
# EXPERIMENTAL THRESHOLDS - Baseados em análise ROC
# =============================================================================

SMART_CONSERVATIVE = 12       # Alta precisão (baixo falso positivo)
SMART_BALANCED = 909         # Melhor F1 score (experimental - muito alto)
SMART_AGGRESSIVE = 909       # Alto recall (experimental - muito alto)

# =============================================================================
# STRATEGY MAPPING
# =============================================================================

STRATEGY_MAP = {
    # Practical strategies (RECOMMENDED)
    'conservative_practical': CONSERVATIVE_PRACTICAL,
    'moderate_practical': MODERATE_PRACTICAL,
    'quality_focused': QUALITY_FOCUSED,
    
    # Generic strategies
    'conservative': CONSERVATIVE_THRESHOLD,
    'balanced': BALANCED_THRESHOLD,
    'aggressive': AGGRESSIVE_THRESHOLD,
    'very_aggressive': VERY_AGGRESSIVE_THRESHOLD,
    
    # Custom strategies (based on user data)
    'custom_conservative': CUSTOM_CONSERVATIVE_THRESHOLD,
    'custom_balanced': CUSTOM_BALANCED_THRESHOLD,
    'custom_aggressive': CUSTOM_AGGRESSIVE_THRESHOLD,
    
    # Experimental strategies
    'smart_conservative': SMART_CONSERVATIVE,
    'smart_balanced': SMART_BALANCED,
    'smart_aggressive': SMART_AGGRESSIVE,
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_threshold_by_strategy(strategy: str) -> int:
    """
    Get blur threshold by strategy name
    
    Args:
        strategy: Strategy name (e.g., 'conservative_practical')
        
    Returns:
        Threshold value
    """
    return STRATEGY_MAP.get(strategy, DEFAULT_PRACTICAL_THRESHOLD)


def categorize_blur_level(blur_score: float, threshold: int = 0) -> str:
    """
    Categorize blur level based on score
    
    Args:
        blur_score: Blur score from Variance of Laplacian
        threshold: Optional threshold for custom categorization
        
    Returns:
        Category string
    """
    if threshold and blur_score < threshold:
        return 'blurry'
    
    # Standard categorization
    if blur_score < 30:
        return 'extremely_blurry'
    elif blur_score < 50:
        return 'very_blurry'
    elif blur_score < 78:
        return 'blurry'
    elif blur_score < 150:
        return 'acceptable'
    else:
        return 'sharp'


def get_strategy_info(strategy: str) -> dict:
    """
    Get strategy information
    
    Args:
        strategy: Strategy name
        
    Returns:
        Dictionary with strategy info
    """
    info_map = {
        'conservative_practical': {
            'threshold': CONSERVATIVE_PRACTICAL,
            'description': 'Remove apenas blur extremo - deixa decisões subjetivas para o usuário',
            'removal_rate': '10-15%',
            'use_case': 'Primeira triagem - removes apenas casos óbvios'
        },
        'moderate_practical': {
            'threshold': MODERATE_PRACTICAL,
            'description': 'Threshold moderado - boa separação entre sharp e blur',
            'removal_rate': '25-35%',
            'use_case': 'Triagem média - remove blur claro mas preserva casos limítrofes'
        },
        'quality_focused': {
            'threshold': QUALITY_FOCUSED,
            'description': 'Foco em qualidade técnica - pode rejeitar imagens com valor contextual',
            'removal_rate': '40-60%',
            'use_case': 'Para portfólios onde qualidade técnica é prioridade'
        }
    }
    
    return info_map.get(strategy, {
        'threshold': get_threshold_by_strategy(strategy),
        'description': f'Threshold: {get_threshold_by_strategy(strategy)}',
        'removal_rate': 'Unknown',
        'use_case': 'Generic strategy'
    })


def get_blur_threshold(strategy: str = 'balanced') -> float:
    """
    Get blur threshold for specific strategy.
    
    Args:
        strategy: Strategy name
    
    Returns:
        float: Threshold value
    """
    # Map strategy names to their threshold values
    strategy_map = {
        'conservative': CONSERVATIVE_PRACTICAL,
        'balanced': MODERATE_PRACTICAL,
        'moderate': MODERATE_PRACTICAL,
        'aggressive': QUALITY_FOCUSED,
        'very_aggressive': VERY_AGGRESSIVE_THRESHOLD,
        'default': MODERATE_PRACTICAL
    }
    
    return strategy_map.get(strategy, MODERATE_PRACTICAL)

# =============================================================================
# ANALYSIS RESULTS - Para referência
# =============================================================================

ANALYSIS_RESULTS = {
    'supervised_validation': {
        'samples_analyzed': 385,
        'blur_rejections': 190,
        'quality_images': 195,
        'rejection_patterns': {
            'blur_mean': 261.54,
            'cropped_mean': 63.46,
            'dark_mean': 66.28,
            'light_mean': 192.02,
            'other_mean': 599.04
        }
    },
    'key_insights': [
        'Manual rejections include subjective criteria beyond technical blur',
        'Users accept images with technical blur if they have contextual value',
        'Blur detection should be assistant tool, not absolute decision maker',
        'Conservative approach preserves more user-valued content'
    ],
    'recommendations': {
        'default_strategy': 'conservative_practical',
        'reasoning': 'Best balance between automation and user preference preservation'
    }
}

# ANÁLISE SUPERVISIONADA - ESTATÍSTICAS E INSIGHTS
# Baseado em análise com 440 exemplos rotulados (Data: 2025-06-22)
SUPERVISED_ANALYSIS = {
    'total_samples': 440,
    'rejection_samples': 245,
    'quality_samples': 195,
    'rejection_blur_mean': 185.38,
    'quality_blur_mean': 98.88,
    'best_threshold_accuracy': 0.52,
    'counter_intuitive_finding': True,
    'main_insight': 'Rejected images are sharper than quality ones - blur is not the primary manual rejection criterion',
    'other_rejection_factors': ['composition', 'exposure', 'color', 'content', 'framing']
}

# USE CASE RECOMMENDATIONS
USE_CASE_RECOMMENDATIONS = {
    'conservative': ['personal_archive', 'family_memories', 'historical_preservation'],
    'balanced': ['general_use', 'automatic_curation', 'basic_classification'],
    'aggressive': ['printing', 'exhibition', 'portfolio'],
    'very_aggressive': ['professional_portfolio', 'high_quality_selection', 'strict_curation']
}

def classify_blur_level(blur_score: float, strategy: str = 'balanced') -> Dict[str, Union[str, float]]:
    """
    Classify blur level based on score and strategy with detailed recommendations.
    
    Args:
        blur_score: Blur score of the image
        strategy: Strategy to use for classification
    
    Returns:
        dict: Detailed classification with level, description, and recommendation
    """
    threshold = get_blur_threshold(strategy)
    
    if blur_score < 20:
        level = 'extremely_blurry'
        description = 'Extremely blurry - consider discarding'
        recommendation = 'remove'
    elif blur_score < threshold:
        level = 'blurry'
        description = f'Blurry by {strategy} criteria'
        recommendation = 'review'
    elif blur_score < threshold * 2:
        level = 'acceptable'
        description = 'Acceptable sharpness'
        recommendation = 'keep'
    else:
        level = 'sharp'
        description = 'Sharp image'
        recommendation = 'keep'
    
    return {
        'level': level,
        'description': description,
        'recommendation': recommendation,
        'threshold_used': threshold,
        'strategy_used': strategy,
        'score': blur_score
    }

# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

# For backward compatibility
DEFAULT_BLUR_THRESHOLD = DEFAULT_PRACTICAL_THRESHOLD
RECOMMENDED_THRESHOLD = CONSERVATIVE_PRACTICAL


if __name__ == "__main__":
    print("🎚️  Blur Detection Configuration")
    print("=" * 40)
    print(f"Default strategy: conservative_practical")
    print(f"Default threshold: {DEFAULT_PRACTICAL_THRESHOLD}")
    print("\nAvailable strategies:")
    for strategy, threshold in STRATEGY_MAP.items():
        info = get_strategy_info(strategy)
        print(f"  • {strategy}: {threshold} - {info.get('description', 'N/A')}")
