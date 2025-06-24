#!/usr/bin/env python3
"""
Visualization Tools - Consolidated visualization utilities
Photo Culling System v2.0
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from typing import Dict, List, Tuple, Any, Optional
import sqlite3
from pathlib import Path
import pandas as pd

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from core.feature_extractor import FeatureExtractor
    from core.person_detector import PersonDetector
except ImportError:
    print("Warning: Could not import core modules. Some features may not work.")
    FeatureExtractor = None
    PersonDetector = None

class DetectionVisualizer:
    """Visualize person detection results"""
    
    def __init__(self, input_dir: Optional[str] = None, output_dir: Optional[str] = None):
        self.input_dir = input_dir or "data/input"
        self.output_dir = output_dir or "data/quality/visualizations"
        self.person_detector = PersonDetector() if PersonDetector else None
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
    def visualize_person_detections(self, image_path: str, save_result: bool = True) -> Dict[str, Any]:
        """Visualize person detections on a single image"""
        if not self.person_detector:
            return {'error': 'PersonDetector not available'}
            
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'error': f'Could not load image: {image_path}'}
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect persons
            detection_result = self.person_detector.detect_persons(image)
            
            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Original image
            axes[0].imshow(image_rgb)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Image with detections
            vis_image = image_rgb.copy()
            
            if 'person_boxes' in detection_result:
                for i, box in enumerate(detection_result['person_boxes']):
                    x1, y1, x2, y2 = box
                    
                    # Draw bounding box
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Add confidence score if available
                    if 'confidence_scores' in detection_result and i < len(detection_result['confidence_scores']):
                        conf = detection_result['confidence_scores'][i]
                        cv2.putText(vis_image, f'{conf:.2f}', (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            axes[1].imshow(vis_image)
            axes[1].set_title(f'Detected Persons: {detection_result.get("person_count", 0)}')
            axes[1].axis('off')
            
            # Add detection info
            info_text = f"""
            Person Count: {detection_result.get('person_count', 0)}
            Dominant Person Score: {detection_result.get('dominant_person_score', 0):.3f}
            Detection Method: {detection_result.get('detection_method', 'unknown')}
            """
            
            plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            
            plt.tight_layout()
            
            if save_result:
                filename = Path(image_path).stem + '_detection_viz.png'
                output_path = Path(self.output_dir) / filename
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                return {
                    'status': 'success',
                    'output_path': str(output_path),
                    'detection_result': detection_result
                }
            else:
                plt.show()
                return {
                    'status': 'success',
                    'detection_result': detection_result
                }
                
        except Exception as e:
            return {'error': str(e)}
            
    def visualize_all_detections(self, max_images: int = 20) -> Dict[str, Any]:
        """Create visualization grid for multiple images"""
        if not self.person_detector:
            return {'error': 'PersonDetector not available'}
            
        print(f"🎨 Creating detection visualization grid...")
        
        image_files = list(Path(self.input_dir).glob("*.JPG"))[:max_images]
        
        if not image_files:
            return {'error': 'No images found'}
            
        # Calculate grid size
        cols = min(4, len(image_files))
        rows = (len(image_files) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten() if rows > 1 else [axes]
            
        results = []
        
        for i, image_path in enumerate(image_files):
            try:
                # Load and process image
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                    
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                detection_result = self.person_detector.detect_persons(image)
                
                # Resize for grid display
                height, width = image_rgb.shape[:2]
                max_size = 400
                if max(height, width) > max_size:
                    scale = max_size / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    image_rgb = cv2.resize(image_rgb, (new_width, new_height))
                
                # Draw detections
                if 'person_boxes' in detection_result:
                    for j, box in enumerate(detection_result['person_boxes']):
                        x1, y1, x2, y2 = box
                        # Scale boxes if image was resized
                        if 'scale' in locals():
                            x1, y1, x2, y2 = [int(coord * scale) for coord in [x1, y1, x2, y2]]
                        cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Display in grid
                ax = axes[i] if len(axes) > 1 else axes
                ax.imshow(image_rgb)
                ax.set_title(f'{Path(image_path).stem}\nPersons: {detection_result.get("person_count", 0)}', 
                            fontsize=10)
                ax.axis('off')
                
                results.append({
                    'image': str(image_path),
                    'person_count': detection_result.get('person_count', 0),
                    'dominant_person_score': detection_result.get('dominant_person_score', 0)
                })
                
            except Exception as e:
                if i < len(axes):
                    ax = axes[i] if len(axes) > 1 else axes
                    ax.text(0.5, 0.5, f'Error:\n{str(e)}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{Path(image_path).stem}\nERROR')
                    ax.axis('off')
        
        # Hide unused subplots
        for i in range(len(image_files), len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        
        # Save grid
        output_path = Path(self.output_dir) / 'detection_grid.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return {
            'status': 'success',
            'output_path': str(output_path),
            'processed_images': len(results),
            'results': results
        }


class AnalysisVisualizer:
    """Visualize analysis results and statistics"""
    
    def __init__(self, features_db: Optional[str] = None, output_dir: Optional[str] = None):
        self.features_db = features_db or "data/features/features.db"
        self.output_dir = output_dir or "data/quality/visualizations"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
    def visualize_blur_distribution(self) -> Dict[str, Any]:
        """Create blur score distribution visualization"""
        try:
            conn = sqlite3.connect(self.features_db)
            df = pd.read_sql_query("""
                SELECT filename, sharpness_laplacian, person_count, 
                       brightness_mean, dominant_person_score
                FROM image_features 
                WHERE sharpness_laplacian IS NOT NULL
            """, conn)
            conn.close()
            
            if df.empty:
                return {'error': 'No data found in database'}
                
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Blur score distribution
            axes[0, 0].hist(df['sharpness_laplacian'], bins=50, alpha=0.7, color='blue')
            axes[0, 0].axvline(x=78, color='red', linestyle='--', label='Blur Threshold')
            axes[0, 0].set_xlabel('Sharpness (Laplacian Variance)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Blur Score Distribution')
            axes[0, 0].legend()
            
            # Person count vs blur score
            axes[0, 1].scatter(df['sharpness_laplacian'], df['person_count'], alpha=0.6)
            axes[0, 1].axvline(x=78, color='red', linestyle='--', alpha=0.7)
            axes[0, 1].set_xlabel('Sharpness (Laplacian Variance)')
            axes[0, 1].set_ylabel('Person Count')
            axes[0, 1].set_title('Person Count vs Blur Score')
            
            # Person count distribution
            person_counts = df['person_count'].value_counts().sort_index()
            axes[1, 0].bar(person_counts.index, person_counts.values, alpha=0.7, color='green')
            axes[1, 0].set_xlabel('Person Count')
            axes[1, 0].set_ylabel('Number of Images')
            axes[1, 0].set_title('Person Count Distribution')
            
            # Dominant person score vs blur
            valid_scores = df[df['dominant_person_score'] > 0]
            if not valid_scores.empty:
                axes[1, 1].scatter(valid_scores['sharpness_laplacian'], 
                                  valid_scores['dominant_person_score'], alpha=0.6, color='orange')
                axes[1, 1].axvline(x=78, color='red', linestyle='--', alpha=0.7)
                axes[1, 1].set_xlabel('Sharpness (Laplacian Variance)')
                axes[1, 1].set_ylabel('Dominant Person Score')
                axes[1, 1].set_title('Dominant Person Score vs Blur')
            else:
                axes[1, 1].text(0.5, 0.5, 'No dominant person scores available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Dominant Person Score vs Blur')
            
            plt.tight_layout()
            
            # Save visualization
            output_path = Path(self.output_dir) / 'blur_analysis.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Calculate statistics
            stats = {
                'total_images': len(df),
                'blur_threshold': 78,
                'blurry_images': len(df[df['sharpness_laplacian'] < 78]),
                'sharp_images': len(df[df['sharpness_laplacian'] >= 78]),
                'mean_blur_score': float(df['sharpness_laplacian'].mean()),
                'std_blur_score': float(df['sharpness_laplacian'].std()),
                'images_with_persons': len(df[df['person_count'] > 0]),
                'mean_person_count': float(df['person_count'].mean())
            }
            
            return {
                'status': 'success',
                'output_path': str(output_path),
                'statistics': stats
            }
            
        except Exception as e:
            return {'error': str(e)}


def main():
    """Main visualization function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Photo Culling Visualization Tools')
    parser.add_argument('--detect-single', type=str,
                       help='Visualize person detection for single image')
    parser.add_argument('--detect-grid', action='store_true',
                       help='Create detection grid for multiple images')
    parser.add_argument('--blur-analysis', action='store_true',
                       help='Create blur analysis visualization')
    parser.add_argument('--max-images', type=int, default=20,
                       help='Maximum images for grid visualization')
    parser.add_argument('--output-dir', type=str, default='data/quality/visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    if args.detect_single:
        visualizer = DetectionVisualizer(output_dir=args.output_dir)
        result = visualizer.visualize_person_detections(args.detect_single, save_result=True)
        print(f"Single detection result: {result}")
        
    if args.detect_grid:
        visualizer = DetectionVisualizer(output_dir=args.output_dir)
        result = visualizer.visualize_all_detections(max_images=args.max_images)
        print(f"Grid visualization result: {result}")
        
    if args.blur_analysis:
        visualizer = AnalysisVisualizer(output_dir=args.output_dir)
        result = visualizer.visualize_blur_distribution()
        print(f"Blur analysis result: {result}")
        
    if not any([args.detect_single, args.detect_grid, args.blur_analysis]):
        print("No visualization specified. Use --help for options.")
        

if __name__ == "__main__":
    main()
