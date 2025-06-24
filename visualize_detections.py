#!/usr/bin/env python3
"""
Person Detection Visualizer
Visualizador de Detecção de Pessoas

Creates a visual representation of detected persons without modifying the original image.
Shows bounding boxes, confidence scores, and detection details.
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.append('src')

def visualize_person_detections(image_path: str, output_dir: str = "visualizations"):
    """
    Visualize person detections on an image
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save visualization results
    """
    try:
        # Import modules
        from src.core.feature_extractor import FeatureExtractor
        
        print(f"🔍 Analyzing image: {os.path.basename(image_path)}")
        
        # Load original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Create visualization copy
        vis_image = original_image.copy()
        height, width = vis_image.shape[:2]
        
        # Initialize feature extractor
        extractor = FeatureExtractor()
        
        # Extract features to get person detection data
        features = extractor.extract_features(image_path)
        
        # Parse person detection results
        total_persons = features.get('total_persons', 0)
        dominant_person_bbox = json.loads(features.get('dominant_person_bbox', '[]'))
        dominant_person_score = features.get('dominant_person_score', 0.0)
        person_analysis_data = json.loads(features.get('person_analysis_data', '{}'))
        
        print(f"👥 Total persons detected: {total_persons}")
        print(f"🎯 Dominant person score: {dominant_person_score:.3f}")
        
        # Draw detection results
        if total_persons > 0 and dominant_person_bbox:
            # Draw dominant person bounding box
            x, y, w, h = dominant_person_bbox
            
            # Ensure coordinates are within image bounds
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(1, min(w, width - x))
            h = max(1, min(h, height - y))
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # Draw label
            label = f"Person (Score: {dominant_person_score:.3f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Draw label background
            cv2.rectangle(vis_image, 
                         (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), 
                         (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Draw center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(vis_image, (center_x, center_y), 5, (255, 0, 0), -1)
            
            # Add analysis info
            info_text = []
            if person_analysis_data:
                area_ratio = person_analysis_data.get('area_ratio', 0)
                centrality = person_analysis_data.get('centrality', 0)
                confidence = person_analysis_data.get('confidence', 0)
                
                info_text.extend([
                    f"Area Ratio: {area_ratio:.3f}",
                    f"Centrality: {centrality:.3f}",
                    f"Confidence: {confidence:.3f}"
                ])
            
            # Draw info text
            y_offset = height - 100
            for i, text in enumerate(info_text):
                cv2.putText(vis_image, text, (10, y_offset + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(vis_image, text, (10, y_offset + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Add image info
        image_info = [
            f"Original: {os.path.basename(image_path)}",
            f"Dimensions: {width}x{height}",
            f"Persons: {total_persons}",
            f"Detection: {'MediaPipe' if 'mediapipe' in str(type(extractor.person_detector)).lower() else 'OpenCV'}"
        ]
        
        for i, text in enumerate(image_info):
            cv2.putText(vis_image, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_image, text, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{base_name}_person_detection_{timestamp}.jpg")
        
        cv2.imwrite(output_path, vis_image)
        
        print(f"✅ Visualization saved: {output_path}")
        
        # Save detection data as JSON
        detection_data = {
            "original_image": os.path.basename(image_path),
            "timestamp": datetime.now().isoformat(),
            "dimensions": {"width": width, "height": height},
            "detection_results": {
                "total_persons": total_persons,
                "dominant_person": {
                    "bounding_box": dominant_person_bbox,
                    "score": dominant_person_score,
                    "analysis_data": person_analysis_data
                },
                "detector_type": "MediaPipe" if "mediapipe" in str(type(extractor.person_detector)).lower() else "OpenCV"
            }
        }
        
        json_path = os.path.join(output_dir, f"{base_name}_detection_data_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(detection_data, f, indent=2)
        
        print(f"📊 Detection data saved: {json_path}")
        
        return output_path, json_path
        
    except Exception as e:
        print(f"❌ Error visualizing detections: {e}")
        return None, None

def main():
    """Main function to run visualization"""
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use default test image
        image_path = "data/input/IMG_9959.JPG"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("Usage: python visualize_detections.py [image_path]")
        return
    
    print("🎨 Person Detection Visualizer")
    print("=" * 40)
    
    vis_path, json_path = visualize_person_detections(image_path)
    
    if vis_path:
        print(f"\n🎉 Visualization complete!")
        print(f"🖼️  Image: {vis_path}")
        print(f"📋 Data: {json_path}")
        print(f"\n💡 The original image remains unchanged.")
        print(f"📁 Check the 'visualizations' folder for results.")
    else:
        print(f"\n❌ Visualization failed.")

if __name__ == "__main__":
    main()
