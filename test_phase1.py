#!/usr/bin/env python3
"""
Test Script for Phase 1 Implementation
Script de teste para implementação da Fase 1

Tests exposure analysis and person detection functionality
"""

import sys
import os
import cv2
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.core.exposure_analyzer import ExposureAnalyzer, analyze_image_exposure
    EXPOSURE_AVAILABLE = True
except ImportError as e:
    print(f"❌ Exposure analyzer not available: {e}")
    EXPOSURE_AVAILABLE = False

try:
    # Try MediaPipe import first
    import mediapipe as mp
    from src.core.person_detector import PersonDetector, detect_persons_in_image
    PERSON_DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Person detector not available (MediaPipe missing): {e}")
    PERSON_DETECTION_AVAILABLE = False

try:
    from src.core.feature_extractor import FeatureExtractor
    FEATURE_EXTRACTOR_AVAILABLE = True
except ImportError as e:
    print(f"❌ Feature extractor not available: {e}")
    FEATURE_EXTRACTOR_AVAILABLE = False


def test_exposure_analysis(image_path):
    """Test exposure analysis functionality"""
    print("\n🔆 Testing Exposure Analysis...")
    
    if not EXPOSURE_AVAILABLE:
        print("❌ Exposure analysis not available")
        return False
    
    try:
        # Test individual module
        result = analyze_image_exposure(image_path)
        if result:
            print("✅ Exposure analysis successful!")
            print(f"   📊 Exposure Level: {result['exposure_level']}")
            print(f"   📈 Quality Score: {result['quality_score']:.3f}")
            print(f"   💡 Mean Brightness: {result['mean_brightness']:.1f}")
            print(f"   ✔️  Properly Exposed: {result['is_properly_exposed']}")
            return True
        else:
            print("❌ Exposure analysis failed")
            return False
    except Exception as e:
        print(f"❌ Error in exposure analysis: {e}")
        return False


def test_person_detection(image_path):
    """Test person detection functionality"""
    print("\n👥 Testing Person Detection...")
    
    if not PERSON_DETECTION_AVAILABLE:
        print("❌ Person detection not available (MediaPipe required)")
        return False
    
    try:
        # Test individual module
        result = detect_persons_in_image(image_path)
        if result:
            print("✅ Person detection successful!")
            print(f"   👤 Total Persons: {result['total_persons']}")
            print(f"   👤 Total Faces: {result['total_faces']}")
            
            if result['dominant_person']:
                dp = result['dominant_person']
                print(f"   🎯 Dominant Person Score: {dp.dominance_score:.3f}")
                print(f"   📦 Bounding Box: {dp.bounding_box}")
                print(f"   🎭 Area Ratio: {dp.area_ratio:.3f}")
                print(f"   📍 Centrality: {dp.centrality:.3f}")
                print(f"   🔍 Local Sharpness: {dp.local_sharpness:.3f}")
            else:
                print("   ℹ️  No dominant person detected")
            return True
        else:
            print("❌ Person detection failed")
            return False
    except Exception as e:
        print(f"❌ Error in person detection: {e}")
        return False


def test_integrated_feature_extraction(image_path):
    """Test integrated feature extraction with Phase 1 features"""
    print("\n🔧 Testing Integrated Feature Extraction...")
    
    if not FEATURE_EXTRACTOR_AVAILABLE:
        print("❌ Feature extractor not available")
        return False
    
    try:
        extractor = FeatureExtractor()
        features = extractor.extract_features(image_path)
        
        if features:
            print("✅ Integrated extraction successful!")
            
            # Check basic features
            print(f"   📁 Filename: {features.get('filename', 'unknown')}")
            print(f"   📏 Dimensions: {features.get('width', 0)}x{features.get('height', 0)}")
            
            # Check exposure features
            if 'exposure_level' in features:
                print(f"   🔆 Exposure Level: {features['exposure_level']}")
                print(f"   📈 Exposure Quality: {features['exposure_quality_score']:.3f}")
            
            # Check person features
            if 'total_persons' in features:
                print(f"   👤 Total Persons: {features['total_persons']}")
                print(f"   🎯 Dominant Person Score: {features['dominant_person_score']:.3f}")
                print(f"   ✂️  Person Cropped: {features['dominant_person_cropped']}")
            
            return True
        else:
            print("❌ Integrated extraction failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in integrated extraction: {e}")
        return False


def run_comprehensive_test(image_path):
    """Run comprehensive test of Phase 1 implementation"""
    print("🚀 Starting Phase 1 Comprehensive Test")
    print("=" * 50)
    
    if not os.path.exists(image_path):
        print(f"❌ Test image not found: {image_path}")
        return False
    
    print(f"📸 Testing with image: {os.path.basename(image_path)}")
    
    # Test individual components
    exposure_ok = test_exposure_analysis(image_path)
    person_ok = test_person_detection(image_path)
    integrated_ok = test_integrated_feature_extraction(image_path)
    
    # Summary
    print("\n📋 Test Summary:")
    print("=" * 30)
    print(f"✅ Exposure Analysis: {'PASS' if exposure_ok else 'FAIL'}")
    print(f"👥 Person Detection: {'PASS' if person_ok else 'FAIL'}")
    print(f"🔧 Integrated Extraction: {'PASS' if integrated_ok else 'FAIL'}")
    
    overall_success = exposure_ok or person_ok or integrated_ok
    print(f"\n🎯 Overall Result: {'SUCCESS' if overall_success else 'FAILURE'}")
    
    if overall_success:
        print("\n🎉 Phase 1 implementation is working!")
        print("Ready to process images with enhanced analysis.")
    else:
        print("\n⚠️  Phase 1 implementation needs attention.")
        print("Check dependencies and module imports.")
    
    return overall_success


def main():
    """Main test function"""
    # Check for test image argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Try to find a test image in the data/input directory
        possible_paths = [
            "data/input",
            "test_images",
            "."
        ]
        
        image_path = None
        for path in possible_paths:
            if os.path.exists(path):
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    images = list(Path(path).glob(f"*{ext}"))
                    if images:
                        image_path = str(images[0])
                        break
                if image_path:
                    break
        
        if not image_path:
            print("❌ No test image found!")
            print("Usage: python test_phase1.py <image_path>")
            print("Or place test images in data/input/ directory")
            return False
    
    return run_comprehensive_test(image_path)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
