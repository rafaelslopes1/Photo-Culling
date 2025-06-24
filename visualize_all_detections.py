#!/usr/bin/env python3
"""
Enhanced Person Detection Visualizer
Visualizador Aprimorado de Detecção de Pessoas

Shows ALL detected persons with the dominant one highlighted.
Mostra TODAS as pessoas detectadas com a dominante destacada.
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

def visualize_all_person_detections(image_path: str, output_dir: str = "visualizations"):
    """
    Visualize ALL person detections on an image, highlighting the dominant one
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save visualization results
    """
    try:
        # Import modules
        from src.core.person_detector import PersonDetector
        from src.core.person_detector_simplified import SimplifiedPersonDetector
        
        print(f"🔍 Analyzing image: {os.path.basename(image_path)}")
        
        # Load original image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Create visualization copy
        vis_image = original_image.copy()
        height, width = vis_image.shape[:2]
        
        # Initialize person detector
        try:
            detector = PersonDetector()
            detector_type = "MediaPipe"
        except ImportError:
            detector = SimplifiedPersonDetector()
            detector_type = "OpenCV"
        
        # Get detection results directly from detector
        detection_results = detector.detect_persons_and_faces(original_image)
        
        total_persons = detection_results.get('total_persons', 0)
        persons = detection_results.get('persons', [])
        faces = detection_results.get('faces', [])
        dominant_person = detection_results.get('dominant_person')
        
        print(f"👥 Total persons detected: {total_persons}")
        print(f"👤 Total faces detected: {len(faces)}")
        if dominant_person:
            print(f"🎯 Dominant person score: {dominant_person.dominance_score:.3f}")
        
        # Color scheme
        regular_color = (255, 150, 0)  # Orange for regular persons
        dominant_color = (0, 255, 0)   # Green for dominant person
        face_color = (0, 0, 255)       # Red for faces
        
        # Draw all person detections
        for i, person in enumerate(persons):
            x, y, w, h = person.bounding_box
            
            # Ensure coordinates are within image bounds
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(1, min(w, width - x))
            h = max(1, min(h, height - y))
            
            # Choose color (dominant vs regular)
            is_dominant = (dominant_person and person.person_id == dominant_person.person_id)
            color = dominant_color if is_dominant else regular_color
            thickness = 4 if is_dominant else 2
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, thickness)
            
            # Draw label
            if is_dominant:
                label = f"DOMINANT Person {i+1} (Score: {person.dominance_score:.3f})"
            else:
                label = f"Person {i+1} (Score: {person.dominance_score:.3f})"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(vis_image, 
                         (x, y - label_size[1] - 10), 
                         (x + label_size[0] + 10, y), 
                         color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(vis_image, (center_x, center_y), 8, color, -1)
            cv2.circle(vis_image, (center_x, center_y), 8, (255, 255, 255), 2)
            
            # Draw person info
            info_y = y + h + 20
            if info_y < height - 20:
                info_text = f"Area: {person.area_ratio:.3f} | Central: {person.centrality:.3f}"
                cv2.putText(vis_image, info_text, (x, info_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw face detections (smaller boxes)
        for i, face in enumerate(faces):
            if 'bbox' in face:
                fx, fy, fw, fh = face['bbox']
                
                # Ensure coordinates are within bounds
                fx = max(0, min(fx, width - 1))
                fy = max(0, min(fy, height - 1))
                fw = max(1, min(fw, width - fx))
                fh = max(1, min(fh, height - fy))
                
                # Draw face box
                cv2.rectangle(vis_image, (fx, fy), (fx + fw, fy + fh), face_color, 1)
                
                # Draw face label
                face_label = f"Face {i+1}"
                cv2.putText(vis_image, face_label, (fx, fy - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, face_color, 1)
        
        # Add comprehensive info panel
        info_panel_height = 160
        info_y_start = 20
        
        info_texts = [
            f"Original: {os.path.basename(image_path)}",
            f"Dimensions: {width}x{height}",
            f"Detector: {detector_type}",
            f"Total Persons: {total_persons}",
            f"Total Faces: {len(faces)}",
            f"Dominant Score: {dominant_person.dominance_score:.3f}" if dominant_person else "No dominant person"
        ]
        
        # Draw info background
        cv2.rectangle(vis_image, (10, 10), (400, info_panel_height), (0, 0, 0), -1)
        cv2.rectangle(vis_image, (10, 10), (400, info_panel_height), (255, 255, 255), 2)
        
        for i, text in enumerate(info_texts):
            y_pos = info_y_start + i * 22
            cv2.putText(vis_image, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_image, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Add legend
        legend_y = height - 80
        legend_items = [
            ("Dominant Person", dominant_color),
            ("Other Persons", regular_color), 
            ("Faces", face_color)
        ]
        
        for i, (label, color) in enumerate(legend_items):
            x_pos = 20 + i * 150
            cv2.rectangle(vis_image, (x_pos, legend_y), (x_pos + 20, legend_y + 15), color, -1)
            cv2.putText(vis_image, label, (x_pos + 25, legend_y + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(vis_image, label, (x_pos + 25, legend_y + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{base_name}_all_persons_{timestamp}.jpg")
        
        cv2.imwrite(output_path, vis_image)
        
        print(f"✅ Visualization saved: {output_path}")
        
        # Save detailed detection data as JSON
        detection_data = {
            "original_image": os.path.basename(image_path),
            "timestamp": datetime.now().isoformat(),
            "dimensions": {"width": width, "height": height},
            "detector_type": detector_type,
            "detection_results": {
                "total_persons": total_persons,
                "total_faces": len(faces),
                "persons": [
                    {
                        "id": person.person_id,
                        "bounding_box": list(person.bounding_box),
                        "dominance_score": person.dominance_score,
                        "area_ratio": person.area_ratio,
                        "centrality": person.centrality,
                        "confidence": person.confidence,
                        "is_dominant": (dominant_person and person.person_id == dominant_person.person_id)
                    }
                    for person in persons
                ],
                "faces": faces,
                "dominant_person": {
                    "id": dominant_person.person_id,
                    "dominance_score": dominant_person.dominance_score,
                    "bounding_box": list(dominant_person.bounding_box)
                } if dominant_person else None
            }
        }
        
        json_path = os.path.join(output_dir, f"{base_name}_all_persons_data_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(detection_data, f, indent=2)
        
        print(f"📊 Detection data saved: {json_path}")
        
        return output_path, json_path
        
    except Exception as e:
        print(f"❌ Error visualizing detections: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Main function to run enhanced visualization"""
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Use default test image
        image_path = "data/input/IMG_9959.JPG"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("Usage: python visualize_all_detections.py [image_path]")
        return
    
    print("🎨 Enhanced Person Detection Visualizer")
    print("=" * 45)
    print("🔍 Shows ALL detected persons + highlights dominant one")
    
    vis_path, json_path = visualize_all_person_detections(image_path)
    
    if vis_path:
        print(f"\n🎉 Enhanced visualization complete!")
        print(f"🖼️  Image: {vis_path}")
        print(f"📋 Data: {json_path}")
        print(f"\n💡 The original image remains unchanged.")
        print(f"📁 Check the 'visualizations' folder for results.")
        print(f"🌟 Now shows ALL persons detected, not just the dominant one!")
    else:
        print(f"\n❌ Visualization failed.")

if __name__ == "__main__":
    main()
