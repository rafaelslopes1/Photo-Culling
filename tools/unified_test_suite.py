#!/usr/bin/env python3
"""
Unified Test Suite - Photo Culling System v2.5
Suite de testes unificada e otimizada

Combines all testing functionality into a single, comprehensive test suite
with GPU optimization, quiet logging, and complete system validation.
"""

import os
import sys
import time
import warnings
import numpy as np
from typing import Dict, Any, List, Tuple

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
    
    GPU_ENABLED = gpu_config.get('gpu_enabled', False)
    SYSTEM_INFO = system_info
    
except Exception as e:
    print(f"⚠️ Otimização não disponível: {e}")
    GPU_ENABLED = False
    SYSTEM_INFO = {'is_apple_silicon': False}


class UnifiedTestSuite:
    """
    Comprehensive test suite for Photo Culling System
    Suite de testes abrangente para o sistema de seleção de fotos
    """
    
    def __init__(self, show_gpu_info: bool = True):
        """Initialize test suite with optional GPU info display"""
        self.show_gpu_info = show_gpu_info
        self.test_results = []
        
        if self.show_gpu_info and SYSTEM_INFO.get('is_apple_silicon'):
            self._display_system_info()
    
    def _display_system_info(self):
        """Display system optimization information"""
        print("🚀 SISTEMA OTIMIZADO")
        print("=" * 60)
        if SYSTEM_INFO['is_apple_silicon']:
            print(f"🔥 Chip: Apple {SYSTEM_INFO['chip_type']}")
            print(f"🎮 GPU: {SYSTEM_INFO['gpu_cores']} cores")
            print(f"⚡ CPU: {SYSTEM_INFO['cpu_cores']} cores") 
            print(f"💾 RAM: {SYSTEM_INFO['memory_gb']}GB unificada")
            print(f"🚀 Aceleração: {'GPU (Metal)' if GPU_ENABLED else 'CPU'}")
        print("=" * 60)
    
    def test_system_health(self) -> bool:
        """Test overall system health and dependencies"""
        print("\n🏥 TESTE DE SAÚDE DO SISTEMA")
        print("=" * 60)
        
        try:
            # Check data directories
            data_dirs = {
                'input': 'data/input',
                'features': 'data/features', 
                'labels': 'data/labels',
                'models': 'data/models',
                'quality': 'data/quality'
            }
            
            for name, path in data_dirs.items():
                if os.path.exists(path):
                    count = len(os.listdir(path))
                    print(f"   ✅ data/{name}: {count} arquivos")
                else:
                    print(f"   ❌ data/{name}: não encontrado")
                    return False
            
            # Test core modules import
            try:
                from src.core.feature_extractor import FeatureExtractor
                from src.core.person_detector import PersonDetector
                from src.core.overexposure_analyzer import OverexposureAnalyzer
                from src.core.unified_scoring_system import UnifiedScoringSystem
                print("   ✅ Módulos core importados com sucesso")
                return True
            except Exception as e:
                print(f"   ❌ Erro ao importar módulos: {e}")
                return False
                
        except Exception as e:
            print(f"   ❌ Erro na verificação: {e}")
            return False
    
    def test_person_detection(self) -> bool:
        """Test person detection with performance measurement"""
        print("\n🚶 TESTE DE DETECÇÃO DE PESSOAS")
        print("=" * 60)
        
        try:
            start_time = time.time()
            
            # Import with optimization
            from src.core.person_detector import PersonDetector
            
            init_time = time.time() - start_time
            
            # Test with sample image
            test_image = "data/input/IMG_0001.JPG"
            if not os.path.exists(test_image):
                print("   ❌ Imagem de teste não encontrada")
                return False
            
            # Load image
            import cv2
            image = cv2.imread(test_image)
            if image is None:
                print("   ❌ Erro ao carregar imagem")
                return False
            
            # Initialize detector
            detector = PersonDetector()
            print(f"   ⏱️ Inicialização: {init_time:.3f}s")
            
            # Time the detection
            detection_start = time.time()
            persons = detector.detect_persons_and_faces(image)
            detection_time = time.time() - detection_start
            
            # Process results
            if isinstance(persons, list):
                person_count = len(persons)
                face_count = sum(1 for p in persons if isinstance(p, dict) and len(p.get('face_landmarks', [])) > 0)
            else:
                person_count = 1 if persons else 0
                face_count = 0
            
            print(f"   ⏱️ Detecção: {detection_time:.3f}s")
            print(f"   👥 Pessoas: {person_count}")
            print(f"   👤 Faces: {face_count}")
            print("   ✅ Detecção funcionando corretamente")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Erro na detecção: {e}")
            return False
    
    def test_feature_extraction(self) -> bool:
        """Test feature extraction with Phase 2.5 integration"""
        print("\n📊 TESTE DE EXTRAÇÃO DE FEATURES")
        print("=" * 60)
        
        try:
            from src.core.feature_extractor import FeatureExtractor
            from src.core.overexposure_analyzer import OverexposureAnalyzer
            from src.core.unified_scoring_system import UnifiedScoringSystem
            
            start_time = time.time()
            extractor = FeatureExtractor()
            init_time = time.time() - start_time
            
            print(f"   ⏱️ Inicialização: {init_time:.3f}s")
            
            # Verify analyzers
            analyzer = OverexposureAnalyzer()
            scorer = UnifiedScoringSystem()
            print("   ✅ Analisadores Fase 2.5: Inicializados")
            
            # Test with sample image
            test_image = "data/input/IMG_0001.JPG"
            
            extraction_start = time.time()
            features = extractor.extract_features(test_image)
            extraction_time = time.time() - extraction_start
            
            if features:
                feature_count = len(features)
                
                # Count Phase 2.5 features
                overexposure_features = sum(1 for k in features.keys() if 'overexposure' in k)
                scoring_features = sum(1 for k in features.keys() if any(x in k for x in ['final_score', 'rating', 'ranking']))
                
                print(f"   ⏱️ Extração: {extraction_time:.3f}s")
                print(f"   📊 Features totais: {feature_count}")
                print(f"   🔥 Superexposição: {overexposure_features}")
                print(f"   📈 Scoring: {scoring_features}")
                
                # Show key results
                critical = features.get('overexposure_is_critical', False)
                score = features.get('final_score', 0) * 100 if features.get('final_score') else 0
                rating = features.get('rating', 'N/A')
                
                print(f"   📊 Superexp. crítica: {'SIM' if critical else 'NÃO'}")
                print(f"   📊 Score final: {score:.1f}%")
                print(f"   📊 Rating: {rating}")
                
                return True
            else:
                print("   ❌ Falha na extração")
                return False
                
        except Exception as e:
            print(f"   ❌ Erro na extração: {e}")
            return False
    
    def test_overexposure_analysis(self) -> bool:
        """Test specific overexposure analysis functionality"""
        print("\n🔥 TESTE DE ANÁLISE DE SUPEREXPOSIÇÃO")
        print("=" * 60)
        
        try:
            from src.core.overexposure_analyzer import OverexposureAnalyzer
            
            analyzer = OverexposureAnalyzer()
            
            # Test the known overexposed image
            test_image = "data/input/IMG_0001.JPG"
            
            import cv2
            image = cv2.imread(test_image)
            if image is None:
                print("   ❌ Erro ao carregar imagem")
                return False
            
            # Mock person detection results for testing
            mock_bbox = (500, 300, 600, 800)  # x, y, w, h
            mock_face_landmarks = np.array([(550, 350), (575, 350), (562, 375)])  # Mock face landmarks
            
            result = analyzer.analyze_person_overexposure(mock_bbox, mock_face_landmarks, image)
            
            if result:
                print(f"   📊 Face crítica: {result.get('is_critical', False)}")
                print(f"   📊 Face ratio: {result.get('face_critical_ratio', 0)*100:.1f}%")
                print(f"   📊 Torso ratio: {result.get('torso_critical_ratio', 0)*100:.1f}%")
                print(f"   📊 Dificuldade: {result.get('recovery_difficulty', 'unknown')}")
                print("   ✅ Análise funcionando corretamente")
                
                return True
            else:
                print("   ❌ Falha na análise")
                return False
                
        except Exception as e:
            print(f"   ❌ Erro na análise: {e}")
            return False
    
    def run_all_tests(self, verbose: bool = True) -> Dict[str, bool]:
        """Run all tests and return results"""
        if verbose:
            print("\n🎯 SUITE DE TESTES UNIFICADA - PHOTO CULLING SYSTEM v2.5")
            print("=" * 80)
        
        total_start = time.time()
        
        # Define tests
        tests = [
            ("Sistema Geral", self.test_system_health),
            ("Detecção de Pessoas", self.test_person_detection),
            ("Extração de Features", self.test_feature_extraction),
            ("Análise de Superexposição", self.test_overexposure_analysis)
        ]
        
        results = {}
        
        # Run tests
        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = result
                self.test_results.append((test_name, result))
            except Exception as e:
                if verbose:
                    print(f"❌ ERRO CRÍTICO em {test_name}: {e}")
                results[test_name] = False
                self.test_results.append((test_name, False))
        
        total_time = time.time() - total_start
        
        # Summary
        if verbose:
            self._print_summary(total_time)
        
        return results
    
    def _print_summary(self, total_time: float):
        """Print test results summary"""
        print(f"\n📊 RESUMO DOS TESTES")
        print("=" * 80)
        
        passed = 0
        for test_name, result in self.test_results:
            status = "✅ PASSOU" if result else "❌ FALHOU"
            print(f"   {test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\n⏱️ Tempo total: {total_time:.2f}s")
        print(f"🎯 Resultado: {passed}/{len(self.test_results)} testes passaram")
        
        if passed == len(self.test_results):
            print("🎉 TODOS OS TESTES PASSARAM - SISTEMA OPERACIONAL!")
            if SYSTEM_INFO.get('is_apple_silicon') and GPU_ENABLED:
                print("🚀 Com otimização máxima de GPU!")
        else:
            print("⚠️ ALGUNS TESTES FALHARAM - VERIFICAR LOGS")


def main():
    """Main test execution function"""
    suite = UnifiedTestSuite(show_gpu_info=True)
    results = suite.run_all_tests(verbose=True)
    
    # Return exit code based on results
    all_passed = all(results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
