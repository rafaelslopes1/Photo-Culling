#!/usr/bin/env python3
"""
Image Culling Pipeline

Usage:
    python image_culling.py <input_folder> <output_folder> [--nsfw_model path/to/model.h5]
         [--blur_threshold 100] [--brightness_threshold 50] [--nsfw_threshold 0.7]

This script:
  - Reads images from the input folder.
  - Detects and skips duplicates (using perceptual hashing).
  - Filters out images that are blurry or low in brightness.
  - Optionally, filters out NSFW images (if an NSFW model file is provided).
  - Calculates a quality score (combining sharpness and brightness) for each image.
  - Ranks images and copies them to the output folder (renamed with ranking)
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
    Process all images in the input folder:
      - Skip duplicates
      - Skip images failing NSFW, blurriness, or low brightness checks
      - Compute quality score for remaining images
      - Rank images and copy them to the output folder (renamed with ranking)
    """
    duplicates = find_duplicates(input_folder)
    ranked_images = []

    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        # Only process image files
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue

        # Skip duplicates
        if image_path in duplicates:
            print(f"Skipping duplicate image: {image_path}")
            continue

        # NSFW filtering (if model provided)
        if nsfw_model and is_nsfw(image_path, nsfw_model, nsfw_threshold):
            print(f"Skipping NSFW image: {image_path}")
            continue

        # Check for low quality (blurry or low light)
        if is_blurry(image_path, blur_threshold) or is_low_light(image_path, brightness_threshold):
            print(f"Skipping low-quality image: {image_path}")
            continue

        # If the image passes all filters, calculate its quality score.
        score = quality_score(image_path)
        ranked_images.append((image_path, score))
        print(f"Accepted: {image_path} (Score: {score:.2f})")

    # Sort images by quality score in descending order (best first)
    ranked_images.sort(key=lambda x: x[1], reverse=True)

    # Copy images to output folder with new filenames that reflect their ranking.
    for idx, (img_path, score) in enumerate(ranked_images):
        ext = os.path.splitext(img_path)[1]
        dest_filename = f"{idx + 1:03d}_{score:.2f}{ext}"
        dest_path = os.path.join(output_folder, dest_filename)
        try:
            shutil.copy(img_path, dest_path)
            print(f"Copied {img_path} -> {dest_path}")
        except Exception as e:
            print(f"Error copying {img_path} to {dest_path}: {e}")

    print("\nProcessing complete. Total images selected:", len(ranked_images))


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

