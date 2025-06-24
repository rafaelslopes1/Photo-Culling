#!/usr/bin/env python3
"""
Analysis of Multi-Person Detection Results
Análise dos Resultados de Detecção Multi-Pessoa

Compares the aggressive detection strategies with the current implementation.
"""

import sys
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

# Add src to path
sys.path.append('src')

from core.feature_extractor import FeatureExtractor

def analyze_detection_accuracy():
    """Analyze detection accuracy across different methods"""
    
    print("🔍 Multi-Person Detection Analysis")
    print("=" * 50)
    
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Test images (manually selected for variety)
    test_images = [
        "data/input/IMG_0001.JPG",
        "data/input/IMG_0243.JPG", 
        "data/input/IMG_0252.JPG",
        "data/input/IMG_0285.JPG",
        "data/input/IMG_0304.JPG"
    ]
    
    results = {}
    
    for image_path in test_images:
        if not os.path.exists(image_path):
            print(f"⚠️  Skipping missing image: {image_path}")
            continue
            
        image_name = os.path.basename(image_path)
        print(f"\n📸 Analyzing {image_name}...")
        
        # Load image
        image = cv2.imread(image_path) 
        if image is None:
            print(f"❌ Could not load {image_path}")
            continue
            
        height, width = image.shape[:2]
        
        # Method 1: Current FeatureExtractor
        try:
            features = extractor.extract_features(image_path)
            current_person_count = features.get('person_count', 0)
            current_dominant_score = features.get('dominant_person_score', 0.0)
        except Exception as e:
            print(f"   ❌ FeatureExtractor failed: {e}")
            current_person_count = 0
            current_dominant_score = 0.0
        
        # Method 2: Aggressive OpenCV Face Detection
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        aggressive_faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.02,
            minNeighbors=2,
            minSize=(20, 20),
            maxSize=(800, 800)
        )
        
        # Filter faces by area (remove very small false positives)
        valid_faces = []
        for (x, y, w, h) in aggressive_faces:
            area_ratio = (w * h) / (width * height)
            if area_ratio > 0.0005:  # Minimum area threshold
                valid_faces.append((x, y, w, h, area_ratio))
        
        # Method 3: MediaPipe (if available)
        mediapipe_faces = 0
        try:
            import mediapipe as mp
            
            with mp.solutions.face_detection.FaceDetection(
                model_selection=1, 
                min_detection_confidence=0.1
            ) as face_detection:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results_mp = face_detection.process(rgb_image)
                
                if results_mp.detections:
                    mediapipe_faces = len(results_mp.detections)
        except:
            mediapipe_faces = 0
        
        # Store results
        results[image_name] = {
            'dimensions': f"{width}x{height}",
            'current_implementation': {
                'person_count': current_person_count,
                'dominant_score': current_dominant_score
            },
            'aggressive_opencv': {
                'total_faces': len(aggressive_faces),
                'valid_faces': len(valid_faces),
                'largest_face_area': max([area for *_, area in valid_faces]) if valid_faces else 0
            },
            'mediapipe': {
                'faces': mediapipe_faces
            }
        }
        
        # Print detailed results
        print(f"   📐 Dimensions: {width}x{height}")
        print(f"   🔄 Current Implementation:")
        print(f"      Person Count: {current_person_count}")
        print(f"      Dominant Score: {current_dominant_score:.3f}")
        print(f"   🎯 Aggressive OpenCV:")
        print(f"      Total Faces: {len(aggressive_faces)}")
        print(f"      Valid Faces: {len(valid_faces)}")
        if valid_faces:
            print(f"      Largest Face Area: {max([area for *_, area in valid_faces]):.4f}")
        print(f"   🤖 MediaPipe:")
        print(f"      Faces: {mediapipe_faces}")
    
    # Summary analysis
    print(f"\n📊 SUMMARY ANALYSIS")
    print("=" * 30)
    
    total_images = len(results)
    current_avg_count = sum(r['current_implementation']['person_count'] for r in results.values()) / total_images
    opencv_avg_count = sum(r['aggressive_opencv']['valid_faces'] for r in results.values()) / total_images
    mediapipe_avg_count = sum(r['mediapipe']['faces'] for r in results.values()) / total_images
    
    print(f"Images Analyzed: {total_images}")
    print(f"Average Person Count:")
    print(f"  Current Implementation: {current_avg_count:.2f}")
    print(f"  Aggressive OpenCV: {opencv_avg_count:.2f}")
    print(f"  MediaPipe: {mediapipe_avg_count:.2f}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("=" * 20)
    
    if opencv_avg_count > current_avg_count * 1.5:
        print("✅ Aggressive OpenCV detects significantly more faces")
        print("   Consider lowering minNeighbors parameter in current implementation")
        
    if mediapipe_avg_count > current_avg_count:
        print("✅ MediaPipe shows better detection rates")
        print("   Consider lowering min_detection_confidence in current implementation")
    
    if current_avg_count > 0:
        print("✅ Current implementation is working")
        print("   Fine-tuning parameters may improve multi-person detection") 
    else:
        print("❌ Current implementation may need parameter adjustment")
        print("   Consider reviewing detection thresholds")

def main():
    """Main function"""
    print("🧪 Multi-Person Detection Analysis Suite")
    print("=" * 45)
    
    analyze_detection_accuracy()

if __name__ == "__main__":
    main()
