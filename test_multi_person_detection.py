#!/usr/bin/env python3
"""
Multi-Person Detection Test
Teste de Detecção de Múltiplas Pessoas

Tests aggressive multi-person detection strategies.
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_aggressive_person_detection(image_path: str):
    """Test aggressive person detection strategies"""
    
    print(f"🔍 Testing aggressive multi-person detection on: {os.path.basename(image_path)}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    height, width = image.shape[:2]
    print(f"📏 Image dimensions: {width}x{height}")
    
    # Test 1: Face detection with lower thresholds
    print("\n1️⃣ Testing OpenCV face detection (aggressive settings)...")
    
    # Load face cascade
    try:
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Very aggressive face detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.02,    # Very fine scaling
            minNeighbors=2,      # Low neighbor requirement
            minSize=(20, 20),    # Very small minimum
            maxSize=(800, 800),  # Large maximum
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        print(f"   👤 Detected {len(faces)} faces with aggressive settings")
        
        for i, (x, y, w, h) in enumerate(faces):
            area_ratio = (w * h) / (width * height)
            print(f"   Face {i+1}: bbox=({x},{y},{w},{h}), area_ratio={area_ratio:.4f}")
            
    except Exception as e:
        print(f"   ❌ Face detection failed: {e}")
    
    # Test 2: MediaPipe with multiple detection attempts
    print("\n2️⃣ Testing MediaPipe with multiple configurations...")
    
    try:
        import mediapipe as mp
        
        # Try different model configurations
        configs = [
            {"model_selection": 0, "min_detection_confidence": 0.1},
            {"model_selection": 0, "min_detection_confidence": 0.3},
            {"model_selection": 1, "min_detection_confidence": 0.1},
            {"model_selection": 1, "min_detection_confidence": 0.3},
        ]
        
        # MediaPipe face detection
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        all_faces = []
        
        for i, config in enumerate(configs):
            print(f"   Testing config {i+1}: {config}")
            
            with mp.solutions.face_detection.FaceDetection(**config) as face_detection:
                results = face_detection.process(rgb_image)
                
                if results.detections:
                    print(f"     Found {len(results.detections)} faces")
                    
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * width)
                        y = int(bbox.ymin * height)
                        w = int(bbox.width * width)
                        h = int(bbox.height * height)
                        confidence = detection.score[0]
                        
                        face_info = {
                            'bbox': (x, y, w, h),
                            'confidence': confidence,
                            'config': i
                        }
                        all_faces.append(face_info)
                else:
                    print(f"     No faces found")
        
        # Remove duplicates and show unique faces
        unique_faces = []
        for face in all_faces:
            is_duplicate = False
            for unique_face in unique_faces:
                # Check if faces overlap significantly
                x1, y1, w1, h1 = face['bbox']
                x2, y2, w2, h2 = unique_face['bbox']
                
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                area1 = w1 * h1
                area2 = w2 * h2
                min_area = min(area1, area2)
                
                if overlap_area > 0.5 * min_area:  # 50% overlap threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_faces.append(face)
        
        print(f"\n   📊 Summary: {len(unique_faces)} unique faces found across all configs")
        for i, face in enumerate(unique_faces):
            x, y, w, h = face['bbox']
            conf = face['confidence']
            area_ratio = (w * h) / (width * height)
            print(f"   Face {i+1}: bbox=({x},{y},{w},{h}), conf={conf:.3f}, area={area_ratio:.4f}")
            
    except ImportError:
        print("   ⚠️  MediaPipe not available")
    except Exception as e:
        print(f"   ❌ MediaPipe test failed: {e}")
    
    # Test 3: OpenCV DNN Face Detection
    print("\n3️⃣ Testing OpenCV DNN face detection...")
    
    try:
        # We'll use the pre-trained model from OpenCV
        # This is more robust than Haar cascades
        print("   Attempting to use OpenCV DNN for face detection...")
        
        # Try to load an additional cascade for profile faces
        profile_cascade_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
        if os.path.exists(profile_cascade_path):
            profile_cascade = cv2.CascadeClassifier(profile_cascade_path)
            profile_faces = profile_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(30, 30)
            )
            print(f"   👤 Profile faces detected: {len(profile_faces)}")
        else:
            print("   ⚠️  Profile cascade not available")
            
    except Exception as e:
        print(f"   ❌ DNN face detection failed: {e}")
    
    print(f"\n✅ Multi-person detection test complete for {os.path.basename(image_path)}")

def main():
    """Main function"""
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "data/input/IMG_9959.JPG"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print("🧪 Multi-Person Detection Test Suite")
    print("=" * 40)
    
    test_aggressive_person_detection(image_path)

if __name__ == "__main__":
    main()
