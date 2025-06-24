#!/usr/bin/env python3
"""
Simple face_recognition test to debug face detection
"""

import face_recognition
import os
from pathlib import Path

def test_face_recognition_direct():
    """Test face_recognition library directly on sample images"""
    
    print("🔍 TESTE DIRETO DO FACE_RECOGNITION")
    print("=" * 50)
    
    # Test images
    input_dir = Path("data/input")
    test_images = ["IMG_0001.JPG", "IMG_0239.JPG", "IMG_0243.JPG"]
    
    for image_name in test_images:
        image_path = input_dir / image_name
        
        if not image_path.exists():
            print(f"❌ {image_name} não encontrado")
            continue
            
        print(f"\n📷 Testando: {image_name}")
        
        try:
            # Load image
            image = face_recognition.load_image_file(str(image_path))
            print(f"   ✅ Imagem carregada: {image.shape}")
            
            # Find faces with different upsampling values
            for upsample in [0, 1, 2]:
                print(f"   🔍 Upsampling = {upsample}")
                face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=upsample)
                print(f"     Faces encontradas: {len(face_locations)}")
                
                if face_locations:
                    for i, location in enumerate(face_locations):
                        top, right, bottom, left = location
                        print(f"     Face {i+1}: ({left}, {top}) -> ({right}, {bottom})")
                
                if len(face_locations) > 0:
                    break  # Found faces, no need to try higher upsampling
                    
        except Exception as e:
            print(f"   ❌ Erro: {e}")

if __name__ == "__main__":
    test_face_recognition_direct()
