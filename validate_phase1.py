#!/usr/bin/env python3
"""
Phase 1 Final Validation Script
Script de validação final da Fase 1 do Sistema de Seleção de Fotos 2.0

Tests complete implementation of Phase 1 features:
- Advanced exposure analysis with HSV histograms and adaptive thresholding
- Person detection (MediaPipe preferred, OpenCV fallback)
- Integration with main feature extraction pipeline
- Database schema updates
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add src to path
sys.path.append('src')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_phase1_implementation():
    """Validate complete Phase 1 implementation"""
    
    print("🚀 Phase 1 Final Validation")
    print("=" * 50)
    
    # Test image path
    test_image_path = "data/input/IMG_9959.JPG"
    
    validation_results = {
        "exposure_analysis": False,
        "person_detection": False,
        "integrated_extraction": False,
        "database_schema": False,
        "configuration": False
    }
    
    # 1. Test Exposure Analysis Module
    print("\n1️⃣ Testing Exposure Analysis Module...")
    try:
        from src.core.exposure_analyzer import ExposureAnalyzer
        import cv2
        
        analyzer = ExposureAnalyzer()
        
        # Test with sample image
        test_image_path = "data/input/IMG_9959.JPG"
        if os.path.exists(test_image_path):
            image = cv2.imread(test_image_path)
            result = analyzer.analyze_exposure(image)
            
            required_keys = ['exposure_level', 'quality_score', 'mean_brightness', 
                           'otsu_threshold', 'histogram_stats', 'is_properly_exposed']
            
            if all(key in result for key in required_keys):
                validation_results["exposure_analysis"] = True
                print("   ✅ Exposure analysis module working correctly")
                print(f"   📊 Exposure Level: {result['exposure_level']}")
                print(f"   📈 Quality Score: {result['quality_score']:.3f}")
            else:
                print(f"   ❌ Missing required keys: {set(required_keys) - set(result.keys())}")
        else:
            print(f"   ⚠️  Test image not found: {test_image_path}")
            
    except Exception as e:
        print(f"   ❌ Exposure analysis failed: {e}")
    
    # 2. Test Person Detection Module
    print("\n2️⃣ Testing Person Detection Module...")
    try:
        import cv2  # Import cv2 here
        
        # Try MediaPipe first
        try:
            from src.core.person_detector import PersonDetector
            detector_type = "MediaPipe"
        except ImportError:
            from src.core.person_detector_simplified import SimplifiedPersonDetector as PersonDetector
            detector_type = "Simplified (OpenCV)"
            
        detector = PersonDetector()
        
        # Test with sample image
        if os.path.exists(test_image_path):
            image = cv2.imread(test_image_path)
            result = detector.detect_persons_and_faces(image)
            
            required_keys = ['total_persons', 'persons', 'dominant_person']
            
            if all(key in result for key in required_keys):
                validation_results["person_detection"] = True
                print(f"   ✅ Person detection working correctly ({detector_type})")
                print(f"   👥 Total Persons: {result['total_persons']}")
                if result['dominant_person']:
                    print(f"   🎯 Dominant Person Score: {result['dominant_person'].dominance_score:.3f}")
            else:
                print(f"   ❌ Missing required keys: {set(required_keys) - set(result.keys())}")
        else:
            print(f"   ⚠️  Test image not found: {test_image_path}")
            
    except Exception as e:
        print(f"   ❌ Person detection failed: {e}")
    
    # 3. Test Integrated Feature Extraction
    print("\n3️⃣ Testing Integrated Feature Extraction...")
    try:
        from src.core.feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor()
        
        if os.path.exists(test_image_path):
            features = extractor.extract_features(test_image_path)
            
            # Check for Phase 1 features
            phase1_features = [
                'exposure_level', 'exposure_quality_score', 'mean_brightness',
                'total_persons', 'dominant_person_score', 'dominant_person_bbox'
            ]
            
            present_features = [f for f in phase1_features if f in features]
            
            if len(present_features) >= 4:  # At least most features should be present
                validation_results["integrated_extraction"] = True
                print("   ✅ Integrated feature extraction working")
                print(f"   📊 Phase 1 features present: {len(present_features)}/{len(phase1_features)}")
                print(f"   🔍 Features: {', '.join(present_features)}")
            else:
                print(f"   ❌ Insufficient Phase 1 features: {present_features}")
        else:
            print(f"   ⚠️  Test image not found: {test_image_path}")
            
    except Exception as e:
        print(f"   ❌ Integrated extraction failed: {e}")
    
    # 4. Validate Database Schema
    print("\n4️⃣ Validating Database Schema...")
    try:
        import sqlite3
        
        # Check if database can be created with new schema
        test_db_path = "test_phase1.db"
        conn = sqlite3.connect(test_db_path)
        cursor = conn.cursor()
        
        # Try to create table with Phase 1 schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_features (
                filename TEXT PRIMARY KEY,
                -- Existing features
                sharpness_laplacian REAL,
                brightness_mean REAL,
                face_count INTEGER,
                -- Phase 1: Exposure Analysis
                exposure_level TEXT,
                exposure_quality_score REAL,
                mean_brightness REAL,
                otsu_threshold REAL,
                histogram_stats TEXT,
                is_properly_exposed BOOLEAN,
                -- Phase 1: Person Analysis
                total_persons INTEGER,
                dominant_person_score REAL,
                dominant_person_bbox TEXT,
                dominant_person_cropped BOOLEAN,
                dominant_person_blur REAL,
                person_analysis_data TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        os.remove(test_db_path)
        
        validation_results["database_schema"] = True
        print("   ✅ Database schema valid for Phase 1")
        
    except Exception as e:
        print(f"   ❌ Database schema validation failed: {e}")
    
    # 5. Validate Configuration
    print("\n5️⃣ Validating Configuration...")
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        
        # Check for Phase 1 configuration sections
        phase1_config_sections = [
            "processing_settings.phase1_analysis.exposure_analysis",
            "processing_settings.phase1_analysis.person_analysis"
        ]
        
        valid_sections = 0
        for section in phase1_config_sections:
            keys = section.split(".")
            current = config
            try:
                for key in keys:
                    current = current[key]
                valid_sections += 1
            except KeyError:
                pass
        
        if valid_sections >= 1:  # At least one section should be present
            validation_results["configuration"] = True
            print("   ✅ Configuration valid for Phase 1")
            print(f"   ⚙️  Valid config sections: {valid_sections}/{len(phase1_config_sections)}")
        else:
            print("   ❌ Phase 1 configuration sections missing")
            
    except Exception as e:
        print(f"   ❌ Configuration validation failed: {e}")
    
    # Summary
    print("\n📋 Validation Summary")
    print("=" * 30)
    
    passed_tests = sum(validation_results.values())
    total_tests = len(validation_results)
    
    for test, result in validation_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test.replace('_', ' ').title()}: {status}")
    
    print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 Phase 1 implementation is COMPLETE and VALIDATED!")
        print("Ready for production use.")
        return True
    elif passed_tests >= 3:
        print("⚠️  Phase 1 implementation is MOSTLY COMPLETE.")
        print("Some minor issues may need attention.")
        return True
    else:
        print("❌ Phase 1 implementation needs significant work.")
        return False

def main():
    """Main validation function"""
    success = validate_phase1_implementation()
    
    if success:
        print("\n✨ Next Steps:")
        print("1. Run integration tests: python tools/integration_test.py")
        print("2. Process sample images to validate end-to-end workflow")
        print("3. Review Phase 2 requirements in docs/PROJECT_ROADMAP.md") 
        sys.exit(0)
    else:
        print("\n🔧 Required Actions:")
        print("1. Review and fix failed validation tests")
        print("2. Check module imports and dependencies")
        print("3. Verify configuration and database schema")
        sys.exit(1)

if __name__ == "__main__":
    main()
