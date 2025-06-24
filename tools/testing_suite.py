#!/usr/bin/env python3
"""
Testing Suite - Consolidated testing and validation utilities
Photo Culling System v2.0
"""

import os
import sys
import json
import time
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional
import sqlite3
from pathlib import Path
import unittest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from core.feature_extractor import FeatureExtractor
    from core.person_detector import PersonDetector
    from core.exposure_analyzer import ExposureAnalyzer
except ImportError:
    print("Warning: Could not import core modules. Some tests may not work.")
    FeatureExtractor = None
    PersonDetector = None
    ExposureAnalyzer = None


class Phase1Validator:
    """Validate Phase 1 implementation completeness"""
    
    def __init__(self, input_dir: Optional[str] = None, features_db: Optional[str] = None):
        self.input_dir = input_dir or "data/input"
        self.features_db = features_db or "data/features/features.db"
        self.feature_extractor = FeatureExtractor() if FeatureExtractor else None
        self.person_detector = PersonDetector() if PersonDetector else None
        self.exposure_analyzer = ExposureAnalyzer() if ExposureAnalyzer else None
        
    def validate_core_modules(self) -> Dict[str, Any]:
        """Validate that all core modules are functional"""
        print("🔍 Validating core modules...")
        
        results = {
            'feature_extractor': False,
            'person_detector': False,
            'exposure_analyzer': False,
            'errors': []
        }
        
        # Test FeatureExtractor
        if self.feature_extractor:
            try:
                # Test with a sample image
                test_images = list(Path(self.input_dir).glob("*.JPG"))[:1]
                if test_images:
                    features = self.feature_extractor.extract_features(str(test_images[0]))
                    required_features = ['sharpness_laplacian', 'brightness_mean', 'person_count']
                    
                    if all(key in features for key in required_features):
                        results['feature_extractor'] = True
                    else:
                        results['errors'].append("FeatureExtractor missing required features")
                else:
                    results['errors'].append("No test images available")
            except Exception as e:
                results['errors'].append(f"FeatureExtractor error: {e}")
        else:
            results['errors'].append("FeatureExtractor not available")
            
        # Test PersonDetector
        if self.person_detector:
            try:
                test_images = list(Path(self.input_dir).glob("*.JPG"))[:1]
                if test_images:
                    image = cv2.imread(str(test_images[0]))
                    if image is not None:
                        detection_result = self.person_detector.detect_persons(image)
                        required_keys = ['person_count', 'detection_method']
                        
                        if all(key in detection_result for key in required_keys):
                            results['person_detector'] = True
                        else:
                            results['errors'].append("PersonDetector missing required result keys")
                    else:
                        results['errors'].append("Could not load test image")
            except Exception as e:
                results['errors'].append(f"PersonDetector error: {e}")
        else:
            results['errors'].append("PersonDetector not available")
            
        # Test ExposureAnalyzer
        if self.exposure_analyzer:
            try:
                test_images = list(Path(self.input_dir).glob("*.JPG"))[:1]
                if test_images:
                    image = cv2.imread(str(test_images[0]))
                    if image is not None:
                        exposure_result = self.exposure_analyzer.analyze_exposure(image)
                        required_keys = ['brightness_mean', 'exposure_assessment']
                        
                        if all(key in exposure_result for key in required_keys):
                            results['exposure_analyzer'] = True
                        else:
                            results['errors'].append("ExposureAnalyzer missing required result keys")
                    else:
                        results['errors'].append("Could not load test image for exposure analysis")
            except Exception as e:
                results['errors'].append(f"ExposureAnalyzer error: {e}")
        else:
            results['errors'].append("ExposureAnalyzer not available")
            
        return results
        
    def validate_database_integration(self) -> Dict[str, Any]:
        """Validate database schema and data consistency"""
        print("🔍 Validating database integration...")
        
        results = {
            'database_exists': False,
            'schema_valid': False,
            'data_consistent': False,
            'record_count': 0,
            'errors': []
        }
        
        try:
            if not Path(self.features_db).exists():
                results['errors'].append("Features database does not exist")
                return results
                
            results['database_exists'] = True
            
            conn = sqlite3.connect(self.features_db)
            cursor = conn.cursor()
            
            # Check schema
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='image_features'")
            if cursor.fetchone():
                results['schema_valid'] = True
                
                # Check required columns
                cursor.execute("PRAGMA table_info(image_features)")
                columns = [col[1] for col in cursor.fetchall()]
                required_columns = [
                    'filename', 'sharpness_laplacian', 'brightness_mean', 
                    'person_count', 'dominant_person_score'
                ]
                
                if all(col in columns for col in required_columns):
                    # Check data consistency
                    cursor.execute("SELECT COUNT(*) FROM image_features WHERE sharpness_laplacian IS NOT NULL")
                    results['record_count'] = cursor.fetchone()[0]
                    
                    if results['record_count'] > 0:
                        results['data_consistent'] = True
                    else:
                        results['errors'].append("No valid records in database")
                else:
                    missing = [col for col in required_columns if col not in columns]
                    results['errors'].append(f"Missing database columns: {missing}")
            else:
                results['errors'].append("image_features table does not exist")
                
            conn.close()
            
        except Exception as e:
            results['errors'].append(f"Database validation error: {e}")
            
        return results
        
    def validate_processing_pipeline(self, sample_count: int = 5) -> Dict[str, Any]:
        """Validate the complete processing pipeline"""
        print(f"🔍 Validating processing pipeline with {sample_count} samples...")
        
        results = {
            'total_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'average_processing_time': 0.0,
            'feature_completeness': {},
            'errors': []
        }
        
        if not self.feature_extractor:
            results['errors'].append("FeatureExtractor not available")
            return results
            
        image_files = list(Path(self.input_dir).glob("*.JPG"))[:sample_count]
        
        if not image_files:
            results['errors'].append("No test images available")
            return results
            
        processing_times = []
        feature_counts = {}
        
        for image_path in image_files:
            try:
                start_time = time.time()
                
                # Extract features
                features = self.feature_extractor.extract_features(str(image_path))
                
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                results['total_processed'] += 1
                results['successful_extractions'] += 1
                
                # Track feature completeness
                for key, value in features.items():
                    if key not in feature_counts:
                        feature_counts[key] = 0
                    if value is not None and value != 0:
                        feature_counts[key] += 1
                        
            except Exception as e:
                results['failed_extractions'] += 1
                results['errors'].append(f"Failed to process {image_path}: {e}")
                
        if processing_times:
            results['average_processing_time'] = sum(processing_times) / len(processing_times)
            
        # Calculate feature completeness percentages
        total_samples = results['successful_extractions']
        if total_samples > 0:
            results['feature_completeness'] = {
                key: (count / total_samples) * 100 
                for key, count in feature_counts.items()
            }
            
        return results
        
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete Phase 1 validation"""
        print("🚀 Running full Phase 1 validation...")
        
        validation_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'core_modules': self.validate_core_modules(),
            'database_integration': self.validate_database_integration(),
            'processing_pipeline': self.validate_processing_pipeline(),
            'overall_status': 'UNKNOWN'
        }
        
        # Determine overall status
        core_ok = all(validation_results['core_modules'][key] for key in 
                     ['feature_extractor', 'person_detector', 'exposure_analyzer'])
        db_ok = (validation_results['database_integration']['database_exists'] and 
                validation_results['database_integration']['schema_valid'] and
                validation_results['database_integration']['data_consistent'])
        pipeline_ok = (validation_results['processing_pipeline']['successful_extractions'] > 0 and
                      validation_results['processing_pipeline']['failed_extractions'] == 0)
        
        if core_ok and db_ok and pipeline_ok:
            validation_results['overall_status'] = 'PASS'
        elif core_ok and db_ok:
            validation_results['overall_status'] = 'PARTIAL'
        else:
            validation_results['overall_status'] = 'FAIL'
            
        return validation_results


class MultiPersonDetectionTester:
    """Test multi-person detection functionality"""
    
    def __init__(self, input_dir: Optional[str] = None):
        self.input_dir = input_dir or "data/input"
        self.person_detector = PersonDetector() if PersonDetector else None
        
    def test_detection_methods(self) -> Dict[str, Any]:
        """Test different detection methods"""
        print("🔍 Testing multi-person detection methods...")
        
        if not self.person_detector:
            return {'error': 'PersonDetector not available'}
            
        results = {
            'mediapipe_tests': 0,
            'opencv_tests': 0,
            'total_tests': 0,
            'detection_accuracy': {},
            'performance_metrics': {},
            'errors': []
        }
        
        test_images = list(Path(self.input_dir).glob("*.JPG"))[:10]
        
        for image_path in test_images:
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                    
                # Test detection
                start_time = time.time()
                detection_result = self.person_detector.detect_persons(image)
                processing_time = time.time() - start_time
                
                results['total_tests'] += 1
                
                # Track detection method used
                method = detection_result.get('detection_method', 'unknown')
                if method == 'mediapipe':
                    results['mediapipe_tests'] += 1
                elif method == 'opencv':
                    results['opencv_tests'] += 1
                    
                # Track performance
                if method not in results['performance_metrics']:
                    results['performance_metrics'][method] = []
                results['performance_metrics'][method].append(processing_time)
                
            except Exception as e:
                results['errors'].append(f"Error testing {image_path}: {e}")
                
        # Calculate averages
        for method, times in results['performance_metrics'].items():
            if times:
                results['performance_metrics'][method] = {
                    'average_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'sample_count': len(times)
                }
                
        return results


def main():
    """Main testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Photo Culling Testing Suite')
    parser.add_argument('--validate-phase1', action='store_true',
                       help='Run full Phase 1 validation')
    parser.add_argument('--test-detection', action='store_true',
                       help='Test multi-person detection')
    parser.add_argument('--output', type=str, default='test_results.json',
                       help='Output file for test results')
    
    args = parser.parse_args()
    
    results = {}
    
    if args.validate_phase1:
        validator = Phase1Validator()
        results['phase1_validation'] = validator.run_full_validation()
        
    if args.test_detection:
        tester = MultiPersonDetectionTester()
        results['detection_tests'] = tester.test_detection_methods()
        
    if not any([args.validate_phase1, args.test_detection]):
        # Run all tests by default
        print("Running all tests...")
        validator = Phase1Validator()
        tester = MultiPersonDetectionTester()
        
        results['phase1_validation'] = validator.run_full_validation()
        results['detection_tests'] = tester.test_detection_methods()
        
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"✅ Testing complete! Results saved to {args.output}")
    
    # Print summary
    if 'phase1_validation' in results:
        pv = results['phase1_validation']
        print(f"\n📊 Phase 1 Validation Summary:")
        print(f"  Overall Status: {pv['overall_status']}")
        print(f"  Core Modules: {pv['core_modules']}")
        print(f"  Database Records: {pv['database_integration']['record_count']}")
        print(f"  Pipeline Success Rate: {pv['processing_pipeline']['successful_extractions']}/{pv['processing_pipeline']['total_processed']}")
        
    if 'detection_tests' in results:
        dt = results['detection_tests']
        print(f"\n📊 Detection Tests Summary:")
        print(f"  Total Tests: {dt['total_tests']}")
        print(f"  MediaPipe Tests: {dt['mediapipe_tests']}")
        print(f"  OpenCV Tests: {dt['opencv_tests']}")


if __name__ == "__main__":
    main()
