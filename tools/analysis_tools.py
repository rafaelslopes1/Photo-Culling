#!/usr/bin/env python3
"""
Analysis Tools - Consolidated analysis and debugging utilities
Photo Culling System v2.0
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
from typing import Dict, List, Tuple, Any
import sqlite3
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.feature_extractor import FeatureExtractor
from core.person_detector import PersonDetector

class MultiPersonAnalyzer:
    """Analyzes multi-person detection results"""
    
    def __init__(self, input_dir: str = None, features_db: str = None):
        self.input_dir = input_dir or "data/input"
        self.features_db = features_db or "data/features/features.db"
        self.feature_extractor = FeatureExtractor()
        self.person_detector = PersonDetector()
        
    def analyze_person_detection_accuracy(self) -> Dict[str, Any]:
        """Analyze person detection accuracy across image dataset"""
        print("🔍 Analyzing multi-person detection accuracy...")
        
        results = {
            'total_images': 0,
            'images_with_persons': 0,
            'images_without_persons': 0,
            'person_count_distribution': {},
            'detection_confidence_stats': {},
            'failed_detections': []
        }
        
        image_files = list(Path(self.input_dir).glob("*.JPG"))
        results['total_images'] = len(image_files)
        
        person_counts = []
        confidences = []
        
        for i, image_path in enumerate(image_files):
            if i % 50 == 0:
                print(f"Processing {i}/{len(image_files)} images...")
                
            try:
                # Load and analyze image
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                    
                # Detect persons
                detection_result = self.person_detector.detect_persons(image)
                person_count = detection_result.get('person_count', 0)
                
                person_counts.append(person_count)
                
                if person_count > 0:
                    results['images_with_persons'] += 1
                    if 'confidence_scores' in detection_result:
                        confidences.extend(detection_result['confidence_scores'])
                else:
                    results['images_without_persons'] += 1
                    
            except Exception as e:
                results['failed_detections'].append({
                    'image': str(image_path),
                    'error': str(e)
                })
        
        # Calculate statistics
        if person_counts:
            unique, counts = np.unique(person_counts, return_counts=True)
            results['person_count_distribution'] = dict(zip(unique.tolist(), counts.tolist()))
            
        if confidences:
            results['detection_confidence_stats'] = {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'median': float(np.median(confidences))
            }
            
        return results
        
    def analyze_blur_rejections(self) -> Dict[str, Any]:
        """Analyze blur detection and rejection patterns"""
        print("🔍 Analyzing blur detection patterns...")
        
        try:
            conn = sqlite3.connect(self.features_db)
            cursor = conn.cursor()
            
            # Get blur scores and rejection data
            cursor.execute("""
                SELECT filename, sharpness_laplacian, person_count, 
                       brightness_mean, dominant_person_score
                FROM image_features 
                WHERE sharpness_laplacian IS NOT NULL
            """)
            
            data = cursor.fetchall()
            conn.close()
            
            if not data:
                return {'error': 'No blur data found in database'}
                
            df = pd.DataFrame(data, columns=[
                'filename', 'sharpness_laplacian', 'person_count',
                'brightness_mean', 'dominant_person_score'
            ])
            
            # Analyze blur patterns
            blur_threshold = 78  # From config
            blurry_images = df[df['sharpness_laplacian'] < blur_threshold]
            sharp_images = df[df['sharpness_laplacian'] >= blur_threshold]
            
            results = {
                'total_images': len(df),
                'blurry_count': len(blurry_images),
                'sharp_count': len(sharp_images),
                'blur_threshold': blur_threshold,
                'blur_stats': {
                    'mean': float(df['sharpness_laplacian'].mean()),
                    'std': float(df['sharpness_laplacian'].std()),
                    'min': float(df['sharpness_laplacian'].min()),
                    'max': float(df['sharpness_laplacian'].max()),
                    'median': float(df['sharpness_laplacian'].median())
                },
                'person_correlation': {
                    'blurry_with_persons': len(blurry_images[blurry_images['person_count'] > 0]),
                    'sharp_with_persons': len(sharp_images[sharp_images['person_count'] > 0]),
                    'avg_persons_blurry': float(blurry_images['person_count'].mean()) if len(blurry_images) > 0 else 0,
                    'avg_persons_sharp': float(sharp_images['person_count'].mean()) if len(sharp_images) > 0 else 0
                }
            }
            
            return results
            
        except Exception as e:
            return {'error': f'Database analysis failed: {e}'}


class SerializationDebugger:
    """Debug serialization issues in feature extraction"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        
    def test_feature_serialization(self, test_image_path: str) -> Dict[str, Any]:
        """Test feature extraction and serialization"""
        print(f"🔧 Testing feature serialization for: {test_image_path}")
        
        try:
            # Load test image
            image = cv2.imread(test_image_path)
            if image is None:
                return {'error': f'Could not load image: {test_image_path}'}
                
            # Extract features
            features = self.feature_extractor.extract_features(test_image_path)
            
            # Test JSON serialization
            try:
                json_str = json.dumps(features, indent=2)
                restored_features = json.loads(json_str)
                
                return {
                    'status': 'success',
                    'original_features': features,
                    'serialized_size': len(json_str),
                    'restoration_successful': features == restored_features,
                    'feature_types': {k: type(v).__name__ for k, v in features.items()}
                }
                
            except Exception as e:
                return {
                    'status': 'serialization_failed',
                    'error': str(e),
                    'features': features,
                    'feature_types': {k: type(v).__name__ for k, v in features.items()}
                }
                
        except Exception as e:
            return {
                'status': 'extraction_failed',
                'error': str(e)
            }


def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Photo Culling Analysis Tools')
    parser.add_argument('--person-analysis', action='store_true', 
                       help='Run person detection analysis')
    parser.add_argument('--blur-analysis', action='store_true',
                       help='Run blur detection analysis')
    parser.add_argument('--serialization-test', type=str,
                       help='Test serialization with specific image path')
    parser.add_argument('--output', type=str, default='analysis_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    results = {}
    analyzer = MultiPersonAnalyzer()
    
    if args.person_analysis:
        results['person_analysis'] = analyzer.analyze_person_detection_accuracy()
        
    if args.blur_analysis:
        results['blur_analysis'] = analyzer.analyze_blur_rejections()
        
    if args.serialization_test:
        debugger = SerializationDebugger()
        results['serialization_test'] = debugger.test_feature_serialization(args.serialization_test)
    
    if not any([args.person_analysis, args.blur_analysis, args.serialization_test]):
        # Run all analyses by default
        print("Running all analyses...")
        results['person_analysis'] = analyzer.analyze_person_detection_accuracy()
        results['blur_analysis'] = analyzer.analyze_blur_rejections()
        
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"✅ Analysis complete! Results saved to {args.output}")
    
    # Print summary
    if 'person_analysis' in results:
        pa = results['person_analysis']
        print(f"\n📊 Person Analysis Summary:")
        print(f"  Total images: {pa['total_images']}")
        print(f"  Images with persons: {pa['images_with_persons']}")
        print(f"  Images without persons: {pa['images_without_persons']}")
        
    if 'blur_analysis' in results:
        ba = results['blur_analysis']
        if 'error' not in ba:
            print(f"\n📊 Blur Analysis Summary:")
            print(f"  Total images: {ba['total_images']}")
            print(f"  Blurry images: {ba['blurry_count']}")
            print(f"  Sharp images: {ba['sharp_count']}")
            print(f"  Mean blur score: {ba['blur_stats']['mean']:.2f}")


if __name__ == "__main__":
    main()
