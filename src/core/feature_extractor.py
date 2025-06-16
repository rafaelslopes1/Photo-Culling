#!/usr/bin/env python3
"""
Consolidated Feature Extractor for Image Classification
Extrator de características consolidado para classificação de imagens
Combina todas as funcionalidades em um sistema unificado
"""

import cv2
import numpy as np
import os
import json
import sqlite3
from datetime import datetime
from PIL import Image, ExifTags
from pathlib import Path
import hashlib
from scipy import stats
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from functools import partial
import time
import warnings
warnings.filterwarnings('ignore')

# Advanced features (optional)
try:
    from skimage import feature, measure, filters, segmentation, exposure
    from skimage.color import rgb2gray, rgb2hsv
    from scipy import ndimage
    from sklearn.preprocessing import StandardScaler
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False
    logging.warning("Advanced features not available. Install scikit-image and scipy.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Sistema consolidado de extração de características para classificação de imagens
    Combina metadados EXIF, análise visual básica e features avançadas
    """
    
    def __init__(self, db_path="data/features/features.db"):
        self.db_path = db_path
        self.init_database()
        self.face_cascade = self._load_face_detector()
        
    def init_database(self):
        """Inicializa banco de dados para features"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_features (
                filename TEXT PRIMARY KEY,
                
                -- Basic metadata
                file_size INTEGER,
                width INTEGER,
                height INTEGER,
                aspect_ratio REAL,
                format TEXT,
                
                -- EXIF data
                camera_make TEXT,
                camera_model TEXT,
                iso INTEGER,
                exposure_time REAL,
                f_number REAL,
                focal_length REAL,
                flash_used BOOLEAN,
                orientation INTEGER,
                datetime_taken TEXT,
                
                -- Technical quality metrics
                sharpness_laplacian REAL,
                sharpness_sobel REAL,
                sharpness_fft REAL,
                brightness_mean REAL,
                brightness_std REAL,
                contrast_rms REAL,
                saturation_mean REAL,
                noise_level REAL,
                
                -- Color analysis
                dominant_colors TEXT,
                color_variance REAL,
                color_temperature REAL,
                
                -- Composition analysis
                rule_of_thirds_score REAL,
                symmetry_score REAL,
                edge_density REAL,
                texture_complexity REAL,
                
                -- Object detection
                face_count INTEGER,
                face_areas TEXT,
                skin_ratio REAL,
                
                -- Advanced features
                visual_complexity REAL,
                aesthetic_score REAL,
                uniqueness_hash TEXT,
                
                -- Extraction metadata
                extraction_timestamp TEXT,
                extraction_version TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def _load_face_detector(self):
        """Carrega detector de faces"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            return cv2.CascadeClassifier(cascade_path)
        except:
            logger.warning("Face detector not available")
            return None
    
    def extract_features(self, image_path):
        """
        Extrai todas as características de uma imagem
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            dict: Dicionário com todas as features extraídas
        """
        try:
            filename = os.path.basename(image_path)
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return None
                
            pil_image = Image.open(image_path)
            
            features = {
                'filename': filename,
                'extraction_timestamp': datetime.now().isoformat(),
                'extraction_version': '2.0_consolidated'
            }
            
            # Basic metadata
            features.update(self._extract_basic_metadata(image_path, pil_image))
            
            # EXIF data
            features.update(self._extract_exif_data(pil_image))
            
            # Technical quality
            features.update(self._extract_quality_metrics(image))
            
            # Color analysis
            features.update(self._extract_color_features(image))
            
            # Composition analysis
            features.update(self._extract_composition_features(image))
            
            # Object detection
            features.update(self._extract_object_features(image))
            
            # Advanced features
            if ADVANCED_FEATURES_AVAILABLE:
                features.update(self._extract_advanced_features(image))
            
            # Hash for uniqueness
            features['uniqueness_hash'] = self._calculate_image_hash(image)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            return None
    
    def _extract_basic_metadata(self, image_path, pil_image):
        """Extrai metadados básicos"""
        features = {}
        
        # File info
        features['file_size'] = os.path.getsize(image_path)
        features['width'] = pil_image.width
        features['height'] = pil_image.height
        features['aspect_ratio'] = pil_image.width / pil_image.height
        features['format'] = pil_image.format
        
        return features
    
    def _extract_exif_data(self, pil_image):
        """Extrai dados EXIF"""
        features = {
            'camera_make': None,
            'camera_model': None,
            'iso': None,
            'exposure_time': None,
            'f_number': None,
            'focal_length': None,
            'flash_used': 0,
            'orientation': 1,
            'datetime_taken': None
        }
        
        try:
            exif = pil_image._getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    
                    if tag == 'Make':
                        features['camera_make'] = str(value)
                    elif tag == 'Model':
                        features['camera_model'] = str(value)
                    elif tag == 'ISOSpeedRatings':
                        features['iso'] = int(value)
                    elif tag == 'ExposureTime':
                        features['exposure_time'] = float(value)
                    elif tag == 'FNumber':
                        features['f_number'] = float(value)
                    elif tag == 'FocalLength':
                        features['focal_length'] = float(value)
                    elif tag == 'Flash':
                        features['flash_used'] = 1 if value else 0
                    elif tag == 'Orientation':
                        features['orientation'] = int(value)
                    elif tag == 'DateTime':
                        features['datetime_taken'] = str(value)
        except:
            pass
            
        return features
    
    def _extract_quality_metrics(self, image):
        """Extrai métricas de qualidade técnica"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Sharpness metrics
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_laplacian = laplacian.var()
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sharpness_sobel = np.sqrt(sobel_x**2 + sobel_y**2).mean()
        
        # FFT-based sharpness
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        sharpness_fft = np.mean(magnitude)
        
        # Brightness and contrast
        brightness_mean = gray.mean()
        brightness_std = gray.std()
        contrast_rms = np.sqrt(np.mean((gray - brightness_mean) ** 2))
        
        # Saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation_mean = hsv[:, :, 1].mean()
        
        # Noise estimation
        noise_level = self._estimate_noise(gray)
        
        return {
            'sharpness_laplacian': float(sharpness_laplacian),
            'sharpness_sobel': float(sharpness_sobel),
            'sharpness_fft': float(sharpness_fft),
            'brightness_mean': float(brightness_mean),
            'brightness_std': float(brightness_std),
            'contrast_rms': float(contrast_rms),
            'saturation_mean': float(saturation_mean),
            'noise_level': float(noise_level)
        }
    
    def _extract_color_features(self, image):
        """Extrai características de cor"""
        # Dominant colors
        pixels = image.reshape(-1, 3)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        dominant_colors = kmeans.cluster_centers_.astype(int).tolist()
        
        # Color variance
        color_variance = np.var(pixels, axis=0).mean()
        
        # Color temperature estimation
        color_temperature = self._estimate_color_temperature(image)
        
        return {
            'dominant_colors': json.dumps(dominant_colors),
            'color_variance': float(color_variance),
            'color_temperature': float(color_temperature)
        }
    
    def _extract_composition_features(self, image):
        """Extrai características de composição"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Rule of thirds
        rule_of_thirds_score = self._calculate_rule_of_thirds(gray)
        
        # Symmetry
        symmetry_score = self._calculate_symmetry(gray)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Texture complexity
        texture_complexity = self._calculate_texture_complexity(gray)
        
        return {
            'rule_of_thirds_score': float(rule_of_thirds_score),
            'symmetry_score': float(symmetry_score),
            'edge_density': float(edge_density),
            'texture_complexity': float(texture_complexity)
        }
    
    def _extract_object_features(self, image):
        """Extrai características de objetos"""
        features = {
            'face_count': 0,
            'face_areas': json.dumps([]),
            'skin_ratio': 0.0
        }
        
        if self.face_cascade is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            features['face_count'] = len(faces)
            
            if len(faces) > 0:
                face_areas = [(w * h) for (x, y, w, h) in faces]
                features['face_areas'] = json.dumps(face_areas)
                
                # Estimate skin ratio
                features['skin_ratio'] = self._estimate_skin_ratio(image, faces)
        
        return features
    
    def _extract_advanced_features(self, image):
        """Extrai características avançadas (se disponível)"""
        if not ADVANCED_FEATURES_AVAILABLE:
            return {'visual_complexity': 0.0, 'aesthetic_score': 0.0}
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Visual complexity
        visual_complexity = self._calculate_visual_complexity(gray)
        
        # Aesthetic score (simplified)
        aesthetic_score = self._calculate_aesthetic_score(image)
        
        return {
            'visual_complexity': float(visual_complexity),
            'aesthetic_score': float(aesthetic_score)
        }
    
    def _estimate_noise(self, gray):
        """Estima nível de ruído"""
        h, w = gray.shape
        m = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
        sigma = np.sum(np.sum(np.absolute(cv2.filter2D(gray, -1, np.array(m)))))
        sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (w-2) * (h-2))
        return sigma
    
    def _estimate_color_temperature(self, image):
        """Estima temperatura de cor"""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        r, g, b = cv2.split(rgb)
        
        # Simplified color temperature estimation
        ratio_rg = np.mean(r) / (np.mean(g) + 1e-6)
        ratio_bg = np.mean(b) / (np.mean(g) + 1e-6)
        
        # Approximate mapping to Kelvin
        if ratio_rg > 1.2:
            return 3000  # Warm
        elif ratio_bg > 1.2:
            return 7000  # Cool
        else:
            return 5000  # Neutral
    
    def _calculate_rule_of_thirds(self, gray):
        """Calcula score da regra dos terços"""
        h, w = gray.shape
        
        # Grid lines
        h_lines = [h // 3, 2 * h // 3]
        v_lines = [w // 3, 2 * w // 3]
        
        # Calculate edge density near grid lines
        edges = cv2.Canny(gray, 50, 150)
        score = 0
        
        for y in h_lines:
            score += np.sum(edges[max(0, y-10):min(h, y+10), :])
        
        for x in v_lines:
            score += np.sum(edges[:, max(0, x-10):min(w, x+10)])
        
        return score / (edges.size + 1e-6)
    
    def _calculate_symmetry(self, gray):
        """Calcula score de simetria"""
        # Horizontal symmetry
        left_half = gray[:, :gray.shape[1]//2]
        right_half = gray[:, gray.shape[1]//2:]
        right_half_flipped = np.fliplr(right_half)
        
        if left_half.shape != right_half_flipped.shape:
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_half_flipped = right_half_flipped[:, :min_width]
        
        diff = np.abs(left_half.astype(float) - right_half_flipped.astype(float))
        symmetry_score = 1.0 - (np.mean(diff) / 255.0)
        
        return symmetry_score
    
    def _calculate_texture_complexity(self, gray):
        """Calcula complexidade de textura"""
        # Local binary pattern
        if ADVANCED_FEATURES_AVAILABLE:
            lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
            return np.var(lbp)
        else:
            # Fallback to simple gradient
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            return np.sqrt(grad_x**2 + grad_y**2).mean()
    
    def _calculate_visual_complexity(self, gray):
        """Calcula complexidade visual"""
        if not ADVANCED_FEATURES_AVAILABLE:
            return 0.0
        
        # Edge density
        edges = feature.canny(gray, sigma=1.0)
        edge_density = np.sum(edges) / edges.size
        
        # Texture measure
        glcm = feature.greycomatrix(gray.astype(np.uint8), [1], [0], 256, symmetric=True, normed=True)
        contrast = feature.greycoprops(glcm, 'contrast')[0, 0]
        
        return edge_density * contrast
    
    def _calculate_aesthetic_score(self, image):
        """Calcula score estético simplificado"""
        # Simplified heuristic based on color harmony and composition
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Color harmony (saturation distribution)
        sat_std = np.std(hsv[:, :, 1])
        
        # Brightness distribution
        val_std = np.std(hsv[:, :, 2])
        
        # Combine metrics
        aesthetic_score = min(1.0, (sat_std + val_std) / 128.0)
        
        return aesthetic_score
    
    def _estimate_skin_ratio(self, image, faces):
        """Estima proporção de pele na imagem"""
        if len(faces) == 0:
            return 0.0
        
        # Simple skin detection based on faces
        total_face_area = sum(w * h for (x, y, w, h) in faces)
        image_area = image.shape[0] * image.shape[1]
        
        return total_face_area / image_area
    
    def _calculate_image_hash(self, image):
        """Calcula hash da imagem para detecção de duplicatas"""
        # Resize to 8x8 and convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (8, 8))
        
        # Calculate perceptual hash
        avg = resized.mean()
        diff = resized > avg
        
        # Convert to hex string
        hash_str = ''.join(['1' if x else '0' for x in diff.flatten()])
        return hex(int(hash_str, 2))[2:]
    
    def extract_batch(self, image_paths, max_workers=None):
        """
        Extrai features de múltiplas imagens em paralelo
        
        Args:
            image_paths: Lista de caminhos de imagens
            max_workers: Número máximo de threads
            
        Returns:
            list: Lista de features extraídas
        """
        if max_workers is None:
            max_workers = min(multiprocessing.cpu_count(), len(image_paths))
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self.extract_features, path): path 
                for path in image_paths
            }
            
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    features = future.result()
                    if features:
                        results.append(features)
                        logger.info(f"Extracted features from {os.path.basename(path)}")
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
        
        return results
    
    def save_features(self, features_list):
        """Salva features no banco de dados"""
        if not features_list:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for features in features_list:
            placeholders = ', '.join(['?' for _ in features])
            columns = ', '.join(features.keys())
            
            cursor.execute(f'''
                INSERT OR REPLACE INTO image_features ({columns})
                VALUES ({placeholders})
            ''', list(features.values()))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved {len(features_list)} feature sets to database")
    
    def get_labeled_features(self, labels_db_path="data/labels/labels.db"):
        """
        Obtém features para imagens que já foram rotuladas
        
        Args:
            labels_db_path: Caminho para o banco de rótulos
            
        Returns:
            pandas.DataFrame: Features das imagens rotuladas
        """
        import pandas as pd
        
        # Get labeled filenames
        labels_conn = sqlite3.connect(labels_db_path)
        labels_df = pd.read_sql_query(
            "SELECT DISTINCT filename FROM labels", 
            labels_conn
        )
        labels_conn.close()
        
        if len(labels_df) == 0:
            logger.warning("No labeled images found")
            return pd.DataFrame()
        
        # Get features for labeled images
        features_conn = sqlite3.connect(self.db_path)
        
        placeholders = ','.join(['?' for _ in labels_df['filename']])
        query = f"""
            SELECT * FROM image_features 
            WHERE filename IN ({placeholders})
        """
        
        features_df = pd.read_sql_query(
            query, 
            features_conn,
            params=labels_df['filename'].tolist()
        )
        features_conn.close()
        
        logger.info(f"Retrieved features for {len(features_df)} labeled images")
        return features_df

# Convenience functions
def extract_features_from_folder(folder_path, output_db=None, max_workers=None):
    """
    Extrai features de todas as imagens em uma pasta
    
    Args:
        folder_path: Caminho da pasta com imagens
        output_db: Caminho do banco de dados de saída
        max_workers: Número de threads para processamento
        
    Returns:
        int: Número de imagens processadas
    """
    if output_db is None:
        output_db = "data/features/features.db"
    
    extractor = FeatureExtractor(output_db)
    
    # Find image files
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_paths = []
    
    for ext in extensions:
        pattern = os.path.join(folder_path, f"*{ext}")
        image_paths.extend(Path(folder_path).glob(f"*{ext}"))
        image_paths.extend(Path(folder_path).glob(f"*{ext.upper()}"))
    
    image_paths = [str(p) for p in image_paths]
    
    if not image_paths:
        logger.warning(f"No images found in {folder_path}")
        return 0
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Extract features
    features_list = extractor.extract_batch(image_paths, max_workers)
    
    # Save to database
    extractor.save_features(features_list)
    
    logger.info(f"Feature extraction completed. Processed {len(features_list)} images.")
    return len(features_list)

if __name__ == "__main__":
    # Example usage
    input_folder = "data/input"
    if os.path.exists(input_folder):
        processed = extract_features_from_folder(input_folder)
        print(f"Processed {processed} images")
    else:
        print(f"Input folder {input_folder} not found")
