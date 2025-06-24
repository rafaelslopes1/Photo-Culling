#!/usr/bin/env python3
"""
Batch Person Detection Visualizer
Visualizador de Detecção de Pessoas em Lote

Tests person detection on multiple images and creates comparative visualizations.
"""

import os
import sys
import glob
from pathlib import Path

# Add src to path
sys.path.append('src')

def test_multiple_images():
    """Test person detection on multiple sample images"""
    
    print("🔍 Testing Person Detection on Multiple Images")
    print("=" * 50)
    
    # Get sample images
    image_patterns = [
        "data/input/IMG_*.JPG",
        "data/input/*.jpg",
        "data/input/*.jpeg",
        "data/input/*.png"
    ]
    
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(pattern))
    
    # Limit to first 5 images for testing
    image_files = sorted(image_files)[:5]
    
    if not image_files:
        print("❌ No images found in data/input/")
        return
    
    from visualize_detections import visualize_person_detections
    
    results = []
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n📸 Processing image {i}/{len(image_files)}: {os.path.basename(image_path)}")
        
        try:
            vis_path, json_path = visualize_person_detections(image_path)
            if vis_path:
                results.append({
                    "original": image_path,
                    "visualization": vis_path,
                    "data": json_path
                })
                print(f"   ✅ Success")
            else:
                print(f"   ❌ Failed")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary
    print(f"\n📋 Processing Summary")
    print("=" * 30)
    print(f"✅ Successful: {len(results)}/{len(image_files)}")
    print(f"📁 Visualizations saved in: visualizations/")
    
    if results:
        print(f"\n🖼️  Generated visualizations:")
        for result in results:
            print(f"   • {os.path.basename(result['visualization'])}")

def main():
    """Main function"""
    
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        test_multiple_images()
    else:
        # Test single image
        from visualize_detections import main as single_main
        single_main()

if __name__ == "__main__":
    main()
