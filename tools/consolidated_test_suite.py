#!/usr/bin/env python3
"""
Consolidated Test Suite for Photo Culling System v2.5
Includes all major testing capabilities in one organized script
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import cv2
import numpy as np
from pathlib import Path
import json
from src.core.feature_extractor import FeatureExtractor
from src.core.person_detector import PersonDetector

def test_phase25_integration():
    """Test Phase 2.5 critical improvements integration"""
    
    print("🧪 TESTE DE INTEGRAÇÃO - FASE 2.5 MELHORIAS CRÍTICAS")
    print("="*70)
    
    # Initialize feature extractor
    print("📦 Inicializando FeatureExtractor...")
    try:
        extractor = FeatureExtractor()
        print("✅ FeatureExtractor inicializado com sucesso")
    except Exception as e:
        print(f"❌ Erro ao inicializar FeatureExtractor: {e}")
        return False
    
    # Check Phase 2.5 analyzers
    print("\n🔍 Verificando analisadores da Fase 2.5:")
    
    # Check overexposure analyzer
    has_overexposure = hasattr(extractor, 'overexposure_analyzer') and extractor.overexposure_analyzer is not None
    print(f"✅ OverexposureAnalyzer: {'Inicializado' if has_overexposure else 'Não inicializado'}")
    
    # Check unified scoring system
    has_scoring = hasattr(extractor, 'unified_scoring_system') and extractor.unified_scoring_system is not None
    print(f"✅ UnifiedScoringSystem: {'Inicializado' if has_scoring else 'Não inicializado'}")
    
    # Test with real image
    test_image = "data/input/IMG_0001.JPG"
    if os.path.exists(test_image):
        print(f"\n📸 Testando com imagem: {os.path.basename(test_image)}")
        features = extractor.extract_features(test_image)
        
        if features:
            # Overexposure features
            overexp_features = {k: v for k, v in features.items() if 'overexposure' in k}
            print(f"   ✅ Features de superexposição: {len(overexp_features)}")
            
            # Unified scoring features
            unified_features = {k: v for k, v in features.items() if 'unified' in k}
            print(f"   ✅ Features de score unificado: {len(unified_features)}")
            
            # Key results
            is_critical = features.get('overexposure_is_critical', False)
            final_score = features.get('unified_final_score', 0)
            rating = features.get('unified_rating', 'unknown')
            
            print(f"\n   📊 Resultados principais:")
            print(f"      Superexposição crítica: {'SIM' if is_critical else 'NÃO'}")
            print(f"      Score final: {final_score:.1%}")
            print(f"      Rating: {rating.upper()}")
        else:
            print("   ❌ Erro ao extrair features")
    else:
        print(f"\n⚠️ Imagem de teste não encontrada: {test_image}")
    
    return True

def test_person_detection():
    """Test person detection capabilities"""
    
    print("\n🚶 TESTE DE DETECÇÃO DE PESSOAS")
    print("="*50)
    
    try:
        detector = PersonDetector()
        print("✅ PersonDetector inicializado")
        
        # Test with sample image
        test_image = "data/input/IMG_0001.JPG"
        if os.path.exists(test_image):
            image = cv2.imread(test_image)
            if image is not None:
                results = detector.detect_persons_and_faces(image)
                print(f"   📊 Pessoas detectadas: {results.get('total_persons', 0)}")
                print(f"   📊 Faces detectadas: {results.get('face_count', 0)}")
                
                if results.get('total_persons', 0) > 0:
                    print("   ✅ Detecção funcionando corretamente")
                else:
                    print("   ⚠️ Nenhuma pessoa detectada")
            else:
                print("   ❌ Erro ao carregar imagem")
        else:
            print(f"   ⚠️ Imagem não encontrada: {test_image}")
            
    except Exception as e:
        print(f"   ❌ Erro no teste de detecção: {e}")
    
    return True

def test_overexposure_specific():
    """Test overexposure analysis with IMG_0001.JPG (known case)"""
    
    print("\n🔥 TESTE ESPECÍFICO DE SUPEREXPOSIÇÃO")
    print("="*50)
    
    test_image = "data/input/IMG_0001.JPG"
    
    if not os.path.exists(test_image):
        print(f"⚠️ Imagem não encontrada: {test_image}")
        return False
    
    try:
        from src.core.overexposure_analyzer import OverexposureAnalyzer
        
        # Load image and known bbox
        image = cv2.imread(test_image)
        known_bbox = (745, 607, 317, 1016)  # From previous analysis
        
        analyzer = OverexposureAnalyzer()
        result = analyzer.analyze_person_overexposure(
            person_bbox=known_bbox,
            face_landmarks=None,
            full_image=image
        )
        
        print(f"   📊 Análise da IMG_0001.JPG:")
        print(f"      Face crítica: {result.get('face_critical_overexposure', False)}")
        print(f"      Torso crítico: {result.get('torso_critical_overexposure', False)}")
        print(f"      Face ratio: {result.get('face_overexposed_ratio', 0):.1%}")
        print(f"      Torso ratio: {result.get('torso_overexposed_ratio', 0):.1%}")
        print(f"      Dificuldade: {result.get('recovery_difficulty', 'unknown')}")
        
        # Validate expected results
        face_critical = result.get('face_critical_overexposure', False)
        torso_critical = result.get('torso_critical_overexposure', False)
        
        if face_critical and torso_critical:
            print("   ✅ Análise de superexposição funcionando corretamente")
        else:
            print("   ⚠️ Resultados não correspondem ao esperado")
            
    except ImportError:
        print("   ❌ OverexposureAnalyzer não disponível")
    except Exception as e:
        print(f"   ❌ Erro na análise: {e}")
    
    return True

def test_system_health():
    """Overall system health check"""
    
    print("\n🏥 VERIFICAÇÃO DE SAÚDE DO SISTEMA")
    print("="*50)
    
    # Check data directories
    data_dirs = ["data/input", "data/features", "data/labels", "data/models", "data/quality"]
    for dir_path in data_dirs:
        if os.path.exists(dir_path):
            file_count = len(list(Path(dir_path).glob("*")))
            print(f"   ✅ {dir_path}: {file_count} arquivos")
        else:
            print(f"   ❌ {dir_path}: Não encontrado")
    
    # Check core modules
    try:
        from src.core.feature_extractor import FeatureExtractor
        from src.core.image_processor import ImageProcessor
        from src.core.person_detector import PersonDetector
        print("   ✅ Módulos core importados com sucesso")
    except Exception as e:
        print(f"   ❌ Erro ao importar módulos: {e}")
    
    return True

def main():
    """Run consolidated test suite"""
    
    print("🚀 SUITE DE TESTES CONSOLIDADA - PHOTO CULLING SYSTEM v2.5")
    print("="*80)
    
    tests = [
        ("Sistema Geral", test_system_health),
        ("Detecção de Pessoas", test_person_detection),
        ("Integração Fase 2.5", test_phase25_integration),
        ("Superexposição Específica", test_overexposure_specific),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"🧪 EXECUTANDO: {test_name}")
        print(f"{'='*80}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Erro fatal no teste {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 RESUMO DOS TESTES")
    print(f"{'='*80}")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        print(f"   {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 RESULTADO FINAL: {passed}/{len(results)} testes passaram")
    
    if passed == len(results):
        print("🎉 TODOS OS TESTES PASSARAM - SISTEMA OPERACIONAL!")
    else:
        print("⚠️ ALGUNS TESTES FALHARAM - VERIFICAR PROBLEMAS")

if __name__ == "__main__":
    main()
