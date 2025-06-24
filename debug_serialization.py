#!/usr/bin/env python3
"""
Debug script to find JSON serialization issues
"""

import cv2
import numpy as np
import json
import sys
import os

# Add src to path
sys.path.append('src')

from src.core.feature_extractor import FeatureExtractor

def main():
    # Load test image
    image_path = "data/input/IMG_9959.JPG"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
        
    # Initialize feature extractor
    extractor = FeatureExtractor()
    
    # Try to extract features
    try:
        features = extractor.extract_features(image_path)
        
        # Try to serialize each feature individually to find the problematic one
        for key, value in features.items():
            try:
                json.dumps(value)
            except TypeError as e:
                print(f"❌ JSON serialization error for '{key}': {e}")
                print(f"   Value: {value} (type: {type(value)})")
                
        print("✅ All features are JSON serializable")
        
    except Exception as e:
        print(f"❌ Error extracting features: {e}")

if __name__ == "__main__":
    main()
