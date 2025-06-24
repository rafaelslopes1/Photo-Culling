#!/usr/bin/env python3
"""
Multi-Person Detection Showcase
Demonstração de Detecção Multi-Pessoa

Visual demonstration of multi-person detection capabilities.
"""

import cv2
import numpy as np
import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.append('src')

from core.feature_extractor import FeatureExtractor

def create_detection_showcase():
    """Create a visual showcase of multi-person detection"""
    
    print("🎨 Multi-Person Detection Showcase")
    print("=" * 40)
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Select diverse test images
    test_images = [
        "data/input/IMG_0001.JPG",  # Expected: 1 person
        "data/input/IMG_0252.JPG",  # Expected: 3 people  
        "data/input/IMG_0304.JPG",  # Expected: 2 people
        "data/input/IMG_0285.JPG",  # Expected: 1 person
        "data/input/IMG_0243.JPG"   # Expected: 1 person
    ]
    
    results = []
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"⚠️  Skipping: {os.path.basename(image_path)} (not found)")
            continue
            
        image_name = os.path.basename(image_path)
        print(f"\n🔍 Processing {image_name}...")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load {image_path}")
            continue
            
        # Extract features including person detection
        try:
            features = extractor.extract_features(image_path)
            
            # Get person analysis results
            person_count = features.get('person_count', 0)
            total_persons = features.get('total_persons', 0)
            dominant_score = features.get('dominant_person_score', 0.0)
            
            # Parse analysis data
            analysis_data = {}
            try:
                analysis_json = features.get('person_analysis_data', '{}')
                analysis_data = json.loads(analysis_json)
            except:
                pass
            
            # Parse dominant person bbox
            dominant_bbox = []
            try:
                bbox_json = features.get('dominant_person_bbox', '[]')
                dominant_bbox = json.loads(bbox_json)
            except:
                pass
            
            result = {
                'image_name': image_name,
                'person_count': person_count,
                'total_persons': total_persons,
                'dominant_score': dominant_score,
                'dominant_bbox': dominant_bbox,
                'analysis_data': analysis_data,
                'dimensions': f"{image.shape[1]}x{image.shape[0]}"
            }
            
            results.append(result)
            
            # Print summary
            print(f"   👥 Detected: {person_count} person(s)")
            print(f"   🏆 Dominant person score: {dominant_score:.3f}")
            if analysis_data:
                print(f"   📊 Area ratio: {analysis_data.get('area_ratio', 0):.4f}")
                print(f"   🎯 Centrality: {analysis_data.get('centrality', 0):.3f}")
                print(f"   💯 Confidence: {analysis_data.get('confidence', 0):.3f}")
            
        except Exception as e:
            print(f"   ❌ Processing failed: {e}")
            continue
    
    # Create summary report
    print(f"\n📈 DETECTION SUMMARY")
    print("=" * 25)
    
    if results:
        total_images = len(results)
        total_people = sum(r['person_count'] for r in results)
        avg_people = total_people / total_images
        avg_score = sum(r['dominant_score'] for r in results) / total_images
        
        print(f"Images processed: {total_images}")
        print(f"Total people detected: {total_people}")
        print(f"Average people per image: {avg_people:.2f}")
        print(f"Average dominant score: {avg_score:.3f}")
        
        # Show individual results
        print(f"\n📸 INDIVIDUAL RESULTS")
        print("-" * 30)
        
        for result in results:
            print(f"{result['image_name']:20} | "
                  f"{result['person_count']:2d} people | "
                  f"Score: {result['dominant_score']:.3f} | "
                  f"{result['dimensions']}")
    
    # Phase 1 validation
    print(f"\n✅ PHASE 1 VALIDATION")
    print("=" * 22)
    
    success_count = sum(1 for r in results if r['person_count'] > 0)
    success_rate = (success_count / len(results)) * 100 if results else 0
    
    print(f"Images with people detected: {success_count}/{len(results)}")
    print(f"Detection success rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 Phase 1 multi-person detection: EXCELLENT")
    elif success_rate >= 60:
        print("✅ Phase 1 multi-person detection: GOOD")
    elif success_rate >= 40:
        print("⚠️  Phase 1 multi-person detection: NEEDS IMPROVEMENT")
    else:
        print("❌ Phase 1 multi-person detection: REQUIRES ATTENTION")
    
    # Save results to JSON
    output_file = "multi_person_detection_results.json"
    try:
        avg_people = sum(r['person_count'] for r in results) / len(results) if results else 0
        avg_score = sum(r['dominant_score'] for r in results) / len(results) if results else 0
        
        with open(output_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_images': len(results),
                    'total_people': sum(r['person_count'] for r in results) if results else 0,
                    'average_people_per_image': avg_people,
                    'average_dominant_score': avg_score,
                    'success_rate': success_rate
                },
                'individual_results': results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")

def main():
    """Main function"""
    
    print("🚀 Photo Culling System v2.0 - Phase 1 Showcase")
    print("=" * 50)
    
    create_detection_showcase()

if __name__ == "__main__":
    main()
