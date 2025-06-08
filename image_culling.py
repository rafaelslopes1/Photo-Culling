#!/usr/bin/env python3
"""
Image Culling Pipeline - Enhanced Version

Usage:
    python image_culling.py <input_folder> <output_folder> [--nsfw_model path/to/model.h5]
         [--blur_threshold 100] [--brightness_threshold 50] [--nsfw_threshold 0.7]

This script processes images and organizes them into categorized folders for easy manual review:

Input Processing:
  - Reads images from the input folder
  - Detects and categorizes duplicates (using perceptual hashing)
  - Filters images based on quality metrics (blur, brightness)
  - Optionally filters NSFW content (if model provided)
  - Calculates quality scores for approved images

Output Organization:
  📁 output/
    ├── selected/     - High-quality images ranked by score (001_85.23_IMG_0001.JPG)
    ├── duplicates/   - Duplicate images detected
    ├── blurry/       - Images that are too blurry
    ├── low_light/    - Images that are too dark
    ├── nsfw/         - NSFW content (if detection enabled)
    └── failed/       - Images that failed processing

This organization allows for easy manual review and recovery of images as needed.
"""

import os
import cv2
import argparse
import shutil
import numpy as np
from PIL import Image
import imagehash

# Try to import the NSFW detection module.
# If not installed or no model provided, we simply skip NSFW filtering.
try:
    from nsfw_detector import predict
except ImportError:
    predict = None
    print("nsfw_detector module not found. NSFW detection will be disabled.")


def find_duplicates(image_folder):
    """
    Find duplicate images in the folder using perceptual hashing.
    Returns a set of duplicate image file paths.
    """
    hash_dict = {}
    duplicates = set()
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        # Only process files with common image extensions
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue
        try:
            img = Image.open(image_path)
            # You can use average_hash, phash, dhash, or whash.
            img_hash = str(imagehash.average_hash(img))
            if img_hash in hash_dict:
                print(f"Duplicate found: {image_path} is similar to {hash_dict[img_hash]}")
                duplicates.add(image_path)
            else:
                hash_dict[img_hash] = image_path
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    return duplicates


def variance_of_laplacian(image_path):
    """
    Compute the Laplacian of the image and return its variance.
    A low variance indicates a blurry image.
    """
    image = cv2.imread(image_path)
    if image is None:
        return 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def is_blurry(image_path, blur_threshold=100):
    """
    Determine if the image is blurry.
    """
    score = variance_of_laplacian(image_path)
    return score < blur_threshold


def is_low_light(image_path, brightness_threshold=50):
    """
    Determine if the image is too dark by computing the average brightness.
    """
    image = cv2.imread(image_path)
    if image is None:
        return True  # If the image cannot be read, treat it as low quality.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2].mean()
    return brightness < brightness_threshold


def is_nsfw(image_path, nsfw_model, nsfw_threshold=0.7):
    """
    Use a NSFW detection model to determine if an image is NSFW.
    Returns True if the NSFW probability exceeds the threshold.
    If no model is provided, returns False.
    """
    if nsfw_model is None:
        return False  # If no NSFW model is provided, skip NSFW filtering.
    try:
        predictions = predict.classify(nsfw_model, image_path)
        # Expecting predictions to be in the form: {image_path: {'nsfw': probability, 'sfw': probability}}
        nsfw_prob = predictions.get(image_path, {}).get('nsfw', 0)
        return nsfw_prob > nsfw_threshold
    except Exception as e:
        print(f"NSFW detection failed for {image_path}: {e}")
        return False


def quality_score(image_path):
    """
    Compute a quality score by combining the sharpness (variance of Laplacian)
    and brightness of the image.
    """
    blur_score = variance_of_laplacian(image_path)
    image = cv2.imread(image_path)
    if image is None:
        return 0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness_score = hsv[:, :, 2].mean()
    # Here, we simply sum the scores. You can adjust weights as needed.
    return blur_score + brightness_score


def process_images(input_folder, output_folder, nsfw_model=None,
                   blur_threshold=100, brightness_threshold=50, nsfw_threshold=0.7):
    """
    Process all images in the input folder and organize them into categorized folders:
      - selected: Images that passed all filters (ranked by quality)
      - duplicates: Duplicate images
      - blurry: Images that are too blurry
      - low_light: Images that are too dark
      - nsfw: NSFW images (if NSFW detection is enabled)
      - failed: Images that could not be processed
    """
    # Create organized output folders
    folders = {
        'selected': os.path.join(output_folder, 'selected'),
        'duplicates': os.path.join(output_folder, 'duplicates'),
        'blurry': os.path.join(output_folder, 'blurry'),
        'low_light': os.path.join(output_folder, 'low_light'),
        'nsfw': os.path.join(output_folder, 'nsfw'),
        'failed': os.path.join(output_folder, 'failed')
    }
    
    for folder_path in folders.values():
        os.makedirs(folder_path, exist_ok=True)
    
    print("Pastas de saída criadas:")
    for category, path in folders.items():
        print(f"  - {category}: {path}")
    print()

    duplicates = find_duplicates(input_folder)
    ranked_images = []
    stats = {
        'total': 0,
        'selected': 0,
        'duplicates': len(duplicates),
        'blurry': 0,
        'low_light': 0,
        'nsfw': 0,
        'failed': 0,
        'skipped': 0
    }

    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        # Only process image files
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            stats['skipped'] += 1
            continue
        
        stats['total'] += 1
        original_filename = os.path.basename(image_path)

        try:
            # Handle duplicates
            if image_path in duplicates:
                dest_path = os.path.join(folders['duplicates'], original_filename)
                shutil.copy(image_path, dest_path)
                print(f"Duplicata movida: {original_filename}")
                continue

            # NSFW filtering (if model provided)
            if nsfw_model and is_nsfw(image_path, nsfw_model, nsfw_threshold):
                dest_path = os.path.join(folders['nsfw'], original_filename)
                shutil.copy(image_path, dest_path)
                stats['nsfw'] += 1
                print(f"NSFW detectado: {original_filename}")
                continue

            # Check for blurriness
            if is_blurry(image_path, blur_threshold):
                dest_path = os.path.join(folders['blurry'], original_filename)
                shutil.copy(image_path, dest_path)
                stats['blurry'] += 1
                print(f"Imagem borrada: {original_filename}")
                continue

            # Check for low light
            if is_low_light(image_path, brightness_threshold):
                dest_path = os.path.join(folders['low_light'], original_filename)
                shutil.copy(image_path, dest_path)
                stats['low_light'] += 1
                print(f"Imagem muito escura: {original_filename}")
                continue

            # If the image passes all filters, calculate its quality score
            score = quality_score(image_path)
            ranked_images.append((image_path, score))
            stats['selected'] += 1
            print(f"Imagem aprovada: {original_filename} (Score: {score:.2f})")

        except Exception as e:
            # Handle images that failed to process
            dest_path = os.path.join(folders['failed'], original_filename)
            try:
                shutil.copy(image_path, dest_path)
                stats['failed'] += 1
                print(f"Erro ao processar (movida para 'failed'): {original_filename} - {e}")
            except Exception as copy_error:
                print(f"Erro crítico ao processar {original_filename}: {e} (Erro de cópia: {copy_error})")

    # Sort selected images by quality score in descending order (best first)
    ranked_images.sort(key=lambda x: x[1], reverse=True)

    # Copy selected images to the selected folder with ranking
    for idx, (img_path, score) in enumerate(ranked_images):
        original_name = os.path.basename(img_path)
        name, ext = os.path.splitext(original_name)
        dest_filename = f"{idx + 1:03d}_{score:.2f}_{name}{ext}"
        dest_path = os.path.join(folders['selected'], dest_filename)
        try:
            shutil.copy(img_path, dest_path)
        except Exception as e:
            print(f"Erro ao copiar {img_path} para pasta 'selected': {e}")

    # Print final statistics
    print("\n" + "="*60)
    print("RELATÓRIO FINAL DE PROCESSAMENTO")
    print("="*60)
    print(f"Total de arquivos processados: {stats['total']}")
    print(f"Arquivos não-imagem ignorados: {stats['skipped']}")
    print(f"")
    print(f"📸 Imagens SELECIONADAS: {stats['selected']}")
    print(f"🔄 Duplicatas encontradas: {stats['duplicates']}")
    print(f"💫 Imagens borradas: {stats['blurry']}")
    print(f"🌑 Imagens muito escuras: {stats['low_light']}")
    if nsfw_model:
        print(f"🔞 Imagens NSFW: {stats['nsfw']}")
    print(f"❌ Falhas no processamento: {stats['failed']}")
    print(f"")
    print("Todas as imagens foram organizadas nas pastas correspondentes para facilitar sua revisão manual!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Image Culling Pipeline")
    parser.add_argument("input_folder", help="Path to the input folder containing images.")
    parser.add_argument("output_folder", help="Path to the output folder for selected images.")
    parser.add_argument("--nsfw_model", help="Path to NSFW model file (optional).")
    parser.add_argument("--blur_threshold", type=float, default=100,
                        help="Threshold for blurriness detection (default: 100).")
    parser.add_argument("--brightness_threshold", type=float, default=50,
                        help="Threshold for brightness (default: 50).")
    parser.add_argument("--nsfw_threshold", type=float, default=0.7,
                        help="Threshold for NSFW detection (default: 0.7).")
    args = parser.parse_args()

    # Verify input folder exists
    if not os.path.exists(args.input_folder):
        print("Error: Input folder does not exist!")
        exit(1)

    # Create output folder if it does not exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        print(f"Created output folder: {args.output_folder}")

    # Load NSFW model if a path is provided and the module is available.
    nsfw_model = None
    if args.nsfw_model:
        if predict is None:
            print("Warning: nsfw_detector module not installed. NSFW filtering will be skipped.")
        else:
            try:
                nsfw_model = predict.load_model(args.nsfw_model)
                print(f"Loaded NSFW model from: {args.nsfw_model}")
            except Exception as e:
                print(f"Failed to load NSFW model: {e}")
                nsfw_model = None

    process_images(args.input_folder, args.output_folder, nsfw_model,
                   blur_threshold=args.blur_threshold,
                   brightness_threshold=args.brightness_threshold,
                   nsfw_threshold=args.nsfw_threshold)


if __name__ == '__main__':
    main()

