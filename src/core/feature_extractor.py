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

# Phase 1 new modules (optional)
try:
    from .exposure_analyzer import ExposureAnalyzer
    from .person_detector import PersonDetector
    PHASE1_FEATURES_AVAILABLE = True
    PERSON_DETECTOR_TYPE = "mediapipe"
except ImportError:
    try:
        from .exposure_analyzer import ExposureAnalyzer
        # from .person_detector_simplified import SimplifiedPersonDetector as PersonDetector
        PHASE1_FEATURES_AVAILABLE = True
        PERSON_DETECTOR_TYPE = "simplified"
        logging.warning("Using simplified person detector (MediaPipe not available)")
    except ImportError:
        PHASE1_FEATURES_AVAILABLE = False
        logging.warning("Phase 1 features not available")

# Phase 2 advanced person analysis (optional)
try:
    from .advanced_person_analyzer import AdvancedPersonAnalyzer
    PHASE2_FEATURES_AVAILABLE = True
    logging.info("Phase 2 advanced person analysis available")
except ImportError:
    PHASE2_FEATURES_AVAILABLE = False
    logging.warning("Phase 2 advanced person analysis not available")

# Phase 2.5 critical improvements (optional)
try:
    from .overexposure_analyzer import OverexposureAnalyzer
    from .unified_scoring_system import UnifiedScoringSystem
    PHASE2_5_FEATURES_AVAILABLE = True
    logging.info("Phase 2.5 critical improvements available")
except ImportError:
    PHASE2_5_FEATURES_AVAILABLE = False
    logging.warning("Phase 2.5 critical improvements not available")

# Phase 3 face recognition (optional)
try:
    from .face_recognition_system import FaceRecognitionSystem
    PHASE3_FEATURES_AVAILABLE = True
    logging.info("Phase 3 face recognition system available")
except ImportError:
    PHASE3_FEATURES_AVAILABLE = False
    logging.warning("Phase 3 face recognition system not available")

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
        
        # Initialize Phase 1 analyzers
        if PHASE1_FEATURES_AVAILABLE:
            try:
                self.exposure_analyzer = ExposureAnalyzer()
                self.person_detector = PersonDetector()
                logging.info(f"Phase 1 analyzers initialized successfully with {PERSON_DETECTOR_TYPE} person detector")
            except Exception as e:
                logging.error(f"Failed to initialize Phase 1 analyzers: {e}")
                self.exposure_analyzer = None
                self.person_detector = None
        else:
            self.exposure_analyzer = None
            self.person_detector = None
            logging.warning("Phase 1 analyzers not available")
        
        # Initialize Phase 2 advanced person analyzer
        if PHASE2_FEATURES_AVAILABLE:
            try:
                self.advanced_person_analyzer = AdvancedPersonAnalyzer()
                logging.info("Phase 2 advanced person analyzer initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize Phase 2 analyzer: {e}")
                self.advanced_person_analyzer = None
        else:
            self.advanced_person_analyzer = None
            logging.warning("Phase 2 advanced person analyzer not available")
        
        # Initialize Phase 2.5 critical improvements
        if PHASE2_5_FEATURES_AVAILABLE:
            try:
                self.overexposure_analyzer = OverexposureAnalyzer()
                self.unified_scoring_system = UnifiedScoringSystem()
                logging.info("Phase 2.5 critical improvements initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize Phase 2.5 analyzers: {e}")
                self.overexposure_analyzer = None
                self.unified_scoring_system = None
        else:
            self.overexposure_analyzer = None
            self.unified_scoring_system = None
            logging.warning("Phase 2.5 critical improvements not available")
        
        # Initialize Phase 3 face recognition system
        if PHASE3_FEATURES_AVAILABLE:
            try:
                self.face_recognition_system = FaceRecognitionSystem()
                logging.info("Phase 3 face recognition system initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize Phase 3 face recognition: {e}")
                self.face_recognition_system = None
        else:
            self.face_recognition_system = None
            logging.warning("Phase 3 face recognition system not available")
        
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
                
                -- Person analysis (Phase 1 - new)
                total_persons INTEGER,
                dominant_person_score REAL,
                dominant_person_bbox TEXT,
                dominant_person_cropped BOOLEAN,
                dominant_person_blur REAL,
                person_analysis_data TEXT,
                
                -- Exposure analysis (Phase 1 - new)
                exposure_level TEXT,
                exposure_quality_score REAL,
                mean_brightness REAL,
                otsu_threshold REAL,
                histogram_stats TEXT,
                is_properly_exposed BOOLEAN,
                
                -- Advanced person analysis (Phase 2 - new)
                -- Person Quality Features
                person_local_blur_score REAL,
                person_lighting_quality REAL,
                person_contrast_score REAL,
                person_relative_sharpness REAL,
                person_quality_score REAL,
                person_quality_level TEXT,
                
                -- Cropping Analysis Features
                has_cropping_issues BOOLEAN,
                cropping_severity TEXT,
                cropping_types TEXT,
                min_edge_distance REAL,
                framing_quality_score REAL,
                composition_score REAL,
                framing_rating TEXT,
                
                -- Pose Quality Features
                posture_quality_score REAL,
                facial_orientation TEXT,
                pose_naturalness TEXT,
                motion_type TEXT,
                pose_stability_score REAL,
                body_symmetry_score REAL,
                pose_rating TEXT,
                
                -- Combined Phase 2 Features
                overall_person_score REAL,
                overall_person_rating TEXT,
                advanced_analysis_version TEXT,
                
                -- Face recognition features (Phase 3 - new)
                face_encodings_count INTEGER,
                face_clusters_found INTEGER,
                similar_faces_count INTEGER,
                face_recognition_data TEXT,
                
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
            
            # Phase 1: Exposure analysis
            if self.exposure_analyzer:
                features.update(self._extract_exposure_features(image))
            
            # Phase 1: Person detection and analysis
            if self.person_detector:
                person_features = self._extract_person_features(image)
                features.update(person_features)
                
                # Add compatibility mapping for person_count
                features['person_count'] = person_features.get('total_persons', 0)
                
                # Phase 3: Face recognition (must be before advanced person analysis)
                if self.face_recognition_system and person_features.get('faces', 0) > 0:
                    face_recognition_features = self._extract_face_recognition_features(image_path)
                    features.update(face_recognition_features)
                
                # Phase 2: Advanced person analysis
                if self.advanced_person_analyzer and person_features.get('total_persons', 0) > 0:
                    advanced_features = self._extract_advanced_person_features(image, person_features)
                    features.update(advanced_features)
                
                # Phase 2.5: Overexposure analysis for person regions
                if self.overexposure_analyzer and person_features.get('total_persons', 0) > 0:
                    overexposure_features = self._extract_overexposure_features(image, person_features)
                    features.update(overexposure_features)
            
            # Advanced features
            if ADVANCED_FEATURES_AVAILABLE:
                features.update(self._extract_advanced_features(image))
            
            # Phase 2.5: Unified scoring system (must be after all other features)
            if self.unified_scoring_system:
                scoring_features = self._extract_unified_scoring_features(features)
                features.update(scoring_features)
            
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
        """Extrai características de objetos usando PersonDetector"""
        features = {
            'face_count': 0,
            'face_areas': json.dumps([]),
            'skin_ratio': 0.0
        }
        
        try:
            # Use PersonDetector instead of face_cascade for better accuracy
            person_result = self.person_detector.detect_persons_and_faces(image)
            faces_data = person_result.get('faces', [])
            
            features['face_count'] = len(faces_data)
            
            if len(faces_data) > 0:
                # Extract face areas from MediaPipe detections
                face_areas = []
                face_regions = []
                
                for face in faces_data:
                    if 'bbox' in face:
                        x, y, w, h = face['bbox']
                        face_areas.append(int(w * h))
                        face_regions.append((x, y, w, h))
                
                features['face_areas'] = json.dumps(face_areas)
                
                # Estimate skin ratio using detected faces
                if face_regions:
                    features['skin_ratio'] = self._estimate_skin_ratio(image, face_regions)
        
        except Exception as e:
            logger.warning(f"Erro na detecção de faces: {e}")
            # Fallback to OpenCV face cascade if available
            if self.face_cascade is not None:
                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    features['face_count'] = int(len(faces))
                    
                    if len(faces) > 0:
                        face_areas = [int(w * h) for (x, y, w, h) in faces]
                        features['face_areas'] = json.dumps(face_areas)
                        features['skin_ratio'] = self._estimate_skin_ratio(image, faces)
                
                except Exception as e2:
                    logger.warning(f"Fallback face detection também falhou: {e2}")
        
        return features
    
    def _extract_exposure_features(self, image):
        """Extract exposure analysis features (Phase 1)"""
        if self.exposure_analyzer is None:
            return {
                'exposure_level': 'adequate',
                'exposure_quality_score': 0.5,
                'mean_brightness': 128.0,
                'otsu_threshold': 128.0,
                'histogram_stats': json.dumps({}),
                'is_properly_exposed': True
            }
        
        try:
            exposure_result = self.exposure_analyzer.analyze_exposure(image)
            
            return {
                'exposure_level': str(exposure_result.get('exposure_level', 'adequate')),
                'exposure_quality_score': float(exposure_result.get('quality_score', 0.5)),
                'mean_brightness': float(exposure_result.get('mean_brightness', 128.0)),
                'otsu_threshold': float(exposure_result.get('otsu_threshold', 128.0)),
                'histogram_stats': json.dumps(exposure_result.get('histogram_stats', {})),
                'is_properly_exposed': bool(exposure_result.get('is_properly_exposed', True))
            }
        except Exception as e:
            logging.error(f"Erro na análise de exposição: {e}")
            return {
                'exposure_level': 'adequate',
                'exposure_quality_score': 0.5,
                'mean_brightness': 128.0,
                'otsu_threshold': 128.0,
                'histogram_stats': json.dumps({}),
                'is_properly_exposed': True
            }
    
    def _extract_person_features(self, image):
        """Extract person detection and analysis features (Phase 1)"""
        if self.person_detector is None:
            return {
                'total_persons': 0,
                'dominant_person_score': 0.0,
                'dominant_person_bbox': json.dumps([]),
                'dominant_person_cropped': False,
                'dominant_person_blur': 0.0,
                'person_analysis_data': json.dumps({})
            }
        
        try:
            person_result = self.person_detector.detect_persons_and_faces(image)
            
            # Initialize default values
            features = {
                'total_persons': person_result.get('total_persons', 0),
                'dominant_person_score': 0.0,
                'dominant_person_bbox': json.dumps([]),
                'dominant_person_cropped': False,
                'dominant_person_blur': 0.0,
                'person_analysis_data': json.dumps({})
            }
            
            # Process dominant person if exists
            dominant_person = person_result.get('dominant_person')
            if dominant_person:
                features['dominant_person_score'] = float(dominant_person.dominance_score)
                features['dominant_person_bbox'] = json.dumps([int(x) for x in dominant_person.bounding_box])
                features['dominant_person_blur'] = float(dominant_person.local_sharpness)
                
                # Check if person is cropped (touches image edges)
                cropping_issues = self._detect_person_cropping(
                    dominant_person.bounding_box, image.shape
                )
                features['dominant_person_cropped'] = bool(len(cropping_issues) > 0)
                
                # Store additional analysis data
                analysis_data = {
                    'area_ratio': float(dominant_person.area_ratio),
                    'centrality': float(dominant_person.centrality),
                    'confidence': float(dominant_person.confidence),
                    'cropping_issues': list(cropping_issues)
                }
                features['person_analysis_data'] = json.dumps(analysis_data)
            
            return features
            
        except Exception as e:
            logging.error(f"Erro na análise de pessoas: {e}")
            return {
                'total_persons': 0,
                'dominant_person_score': 0.0,
                'dominant_person_bbox': json.dumps([]),
                'dominant_person_cropped': False,
                'dominant_person_blur': 0.0,
                'person_analysis_data': json.dumps({})
            }
    
    def _detect_person_cropping(self, bbox, image_shape):
        """Detect if person is cropped at image boundaries"""
        cropping_issues = []
        x, y, w, h = bbox
        img_h, img_w = image_shape[:2]
        tolerance = 10  # pixels
        
        if x <= tolerance:
            cropping_issues.append('left_edge')
        if y <= tolerance:
            cropping_issues.append('top_edge')
        if x + w >= img_w - tolerance:
            cropping_issues.append('right_edge')
        if y + h >= img_h - tolerance:
            cropping_issues.append('bottom_edge')
        
        return cropping_issues
    
    def _extract_advanced_person_features(self, image, person_features):
        """Extract advanced person analysis features (Phase 2)"""
        if self.advanced_person_analyzer is None:
            return {}
        
        try:
            # Get dominant person data from Phase 1
            dominant_person_bbox_str = person_features.get('dominant_person_bbox', '[]')
            person_bbox = json.loads(dominant_person_bbox_str)
            
            if not person_bbox or len(person_bbox) != 4:
                # No valid person bbox, return empty features
                return self._get_empty_advanced_features()
            
            person_bbox = tuple(person_bbox)  # Convert to tuple (x, y, w, h)
            
            # Get analysis data from Phase 1 for additional context
            analysis_data_str = person_features.get('person_analysis_data', '{}')
            dominant_person_data = json.loads(analysis_data_str)
            
            # For now, we don't have face_bbox or pose_landmarks from Phase 1
            # In a full integration, these would be extracted and stored
            face_bbox = None
            pose_landmarks = None
            face_landmarks = None
            
            # Perform comprehensive analysis
            analysis = self.advanced_person_analyzer.analyze_person_comprehensive(
                person_bbox=person_bbox,
                face_bbox=face_bbox,
                pose_landmarks=pose_landmarks,
                face_landmarks=face_landmarks,
                full_image=image,
                dominant_person_data=dominant_person_data
            )
            
            # Extract features for database storage
            advanced_features = self.advanced_person_analyzer.extract_features_for_database(analysis)
            
            return advanced_features
            
        except Exception as e:
            logger.error(f"Erro na extração de features avançadas de pessoa: {e}")
            return self._get_empty_advanced_features()
    
    def _get_empty_advanced_features(self):
        """Return empty advanced features dictionary"""
        return {
            # Person Quality Features
            'person_local_blur_score': 0.0,
            'person_lighting_quality': 0.0,
            'person_contrast_score': 0.0,
            'person_relative_sharpness': 0.0,
            'person_quality_score': 0.0,
            'person_quality_level': 'unknown',
            
            # Cropping Analysis Features
            'has_cropping_issues': False,
            'cropping_severity': 'none',
            'cropping_types': json.dumps([]),
            'min_edge_distance': 0.0,
            'framing_quality_score': 0.0,
            'composition_score': 0.0,
            'framing_rating': 'unknown',
            
            # Pose Quality Features
            'posture_quality_score': 0.0,
            'facial_orientation': 'unknown',
            'pose_naturalness': 'unknown',
            'motion_type': 'unknown',
            'pose_stability_score': 0.0,
            'body_symmetry_score': 0.0,
            'pose_rating': 'unknown',
            
            # Combined Features
            'overall_person_score': 0.0,
            'overall_person_rating': 'unknown',
            'advanced_analysis_version': '2.0'
        }
    
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
        glcm = feature.graycomatrix(gray.astype(np.uint8), [1], [0], 256, symmetric=True, normed=True)
        contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
        
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
    
    def _extract_overexposure_features(self, image, person_features):
        """
        Extract overexposure analysis features for person regions
        
        Args:
            image: OpenCV image
            person_features: Features from person detection including bounding boxes
            
        Returns:
            dict: Overexposure analysis features
        """
        features = {}
        
        try:
            # Get person bounding boxes from person features
            # Try different possible keys for person bounding boxes
            person_bboxes = person_features.get('person_bboxes', [])
            if not person_bboxes:
                # Try to get from dominant_person_bbox
                dominant_bbox = person_features.get('dominant_person_bbox', None)
                if dominant_bbox:
                    # Convert string representation to list if needed
                    if isinstance(dominant_bbox, str):
                        try:
                            import ast
                            dominant_bbox = ast.literal_eval(dominant_bbox)
                        except:
                            logger.warning(f"Could not parse dominant_person_bbox: {dominant_bbox}")
                            dominant_bbox = None
                    
                    if dominant_bbox:
                        person_bboxes = [dominant_bbox]
            
            face_landmarks = person_features.get('face_landmarks', None)
            
            # Debug logging
            logger.info(f"Overexposure analysis: person_bboxes={person_bboxes}, face_landmarks type={type(face_landmarks)}")
            
            if not person_bboxes:
                logger.warning("No person bounding boxes found for overexposure analysis")
                return self._get_default_overexposure_features()
            
            # Analyze overexposure for the first/primary person
            primary_bbox = person_bboxes[0] if person_bboxes else None
            
            if primary_bbox:
                # Convert bbox format if needed (depends on person detector output format)
                if len(primary_bbox) == 4:
                    x, y, w, h = primary_bbox
                    bbox_tuple = (int(x), int(y), int(w), int(h))
                    
                    logger.info(f"Calling overexposure analyzer with bbox: {bbox_tuple}")
                    
                    # Call overexposure analyzer
                    if self.overexposure_analyzer:
                        overexposure_result = self.overexposure_analyzer.analyze_person_overexposure(
                            person_bbox=bbox_tuple,
                            face_landmarks=face_landmarks,
                            full_image=image
                        )
                        logger.info(f"Overexposure result: {overexposure_result}")
                    else:
                        overexposure_result = {}
                
                    # Extract features from result
                    features.update({
                        'overexposure_is_critical': overexposure_result.get('overall_critical_overexposure', False),
                        'overexposure_face_critical_ratio': overexposure_result.get('face_overexposed_ratio', 0.0),
                        'overexposure_face_moderate_ratio': overexposure_result.get('face_overexposed_ratio', 0.0),  # Same value, different interpretation
                        'overexposure_torso_critical_ratio': overexposure_result.get('torso_overexposed_ratio', 0.0),
                        'overexposure_torso_moderate_ratio': overexposure_result.get('torso_overexposed_ratio', 0.0),  # Same value, different interpretation
                        'overexposure_recovery_difficulty': overexposure_result.get('recovery_difficulty', 'unknown'),
                        'overexposure_overall_critical_ratio': overexposure_result.get('max_overexposed_ratio', 0.0),
                        'overexposure_main_reason': overexposure_result.get('main_overexposure_reason', 'unknown'),
                        'overexposure_recommendation': overexposure_result.get('recommendation', 'analyze_manually')
                    })
                else:
                    features = self._get_default_overexposure_features()
            else:
                features = self._get_default_overexposure_features()
                
        except Exception as e:
            logger.error(f"Erro na análise de superexposição: {e}")
            features = self._get_default_overexposure_features()
            
        return features
    
    def _extract_unified_scoring_features(self, all_features):
        """
        Extract unified scoring system features
        
        Args:
            all_features: Dictionary with all previously extracted features
            
        Returns:
            dict: Unified scoring system results
        """
        features = {}
        
        try:
            # Call unified scoring system
            if self.unified_scoring_system:
                scoring_result = self.unified_scoring_system.calculate_final_score(all_features)
            else:
                scoring_result = {}
            
            # Extract features from result
            features.update({
                'unified_final_score': scoring_result.get('final_score', 0.0),
                'unified_rating': scoring_result.get('rating', 'unknown'),
                'unified_is_rejected': scoring_result.get('is_rejected', False),
                'unified_ranking_priority': scoring_result.get('ranking_priority', 0),
                'unified_technical_score': scoring_result.get('score_breakdown', {}).get('technical', 0.0),
                'unified_person_score': scoring_result.get('score_breakdown', {}).get('person', 0.0),
                'unified_composition_score': scoring_result.get('score_breakdown', {}).get('composition', 0.0),
                'unified_context_bonus': scoring_result.get('score_breakdown', {}).get('context_bonus', 0.0),
                'unified_rejection_reasons': str(scoring_result.get('rejection_reasons', [])),
                'unified_main_issues': str([issue.get('type', 'unknown') for issue in scoring_result.get('issues', [])]),
                'unified_recommendation': scoring_result.get('recommendation', 'analyze_manually'),
                'unified_is_recoverable': scoring_result.get('is_recoverable', True)
            })
            
        except Exception as e:
            logger.error(f"Erro no sistema de score unificado: {e}")
            features = self._get_default_scoring_features()
            
        return features
    
    def _get_default_overexposure_features(self):
        """Default overexposure features when analysis is not available"""
        return {
            'overexposure_is_critical': False,
            'overexposure_face_critical_ratio': 0.0,
            'overexposure_face_moderate_ratio': 0.0,
            'overexposure_torso_critical_ratio': 0.0,
            'overexposure_torso_moderate_ratio': 0.0,
            'overexposure_recovery_difficulty': 'unknown',
            'overexposure_overall_critical_ratio': 0.0,
            'overexposure_main_reason': 'analysis_not_available',
            'overexposure_recommendation': 'manual_inspection'
        }
    
    def _get_default_scoring_features(self):
        """Default scoring features when analysis is not available"""
        return {
            'unified_final_score': 50.0,
            'unified_rating': 'unknown',
            'unified_is_rejected': False,
            'unified_ranking_priority': 50,
            'unified_technical_score': 50.0,
            'unified_person_score': 50.0,
            'unified_composition_score': 50.0,
            'unified_context_bonus': 0.0,
            'unified_rejection_reasons': '[]',
            'unified_main_issues': '[]',
            'unified_recommendation': 'system_not_available',
            'unified_is_recoverable': True
        }
    
    def _extract_face_recognition_features(self, image_path):
        """Extract face recognition features (Phase 3)"""
        if self.face_recognition_system is None:
            return {
                'face_encodings_count': 0,
                'face_clusters_found': 0,
                'similar_faces_count': 0,
                'face_recognition_data': json.dumps({})
            }
        
        try:
            # Process image for face recognition
            result = self.face_recognition_system.process_image_for_face_recognition(image_path)
            
            similar_faces = result.get('similar_faces', [])
            
            features = {
                'face_encodings_count': result.get('faces_found', 0),
                'face_clusters_found': 0,  # Will be updated when clustering is performed
                'similar_faces_count': len(similar_faces),
                'face_recognition_data': json.dumps({
                    'status': result.get('status', 'unknown'),
                    'faces_stored': result.get('faces_stored', 0),
                    'similar_faces': similar_faces[:10],  # Store first 10 similar faces
                    'processing_successful': result.get('status') == 'success'
                })
            }
            
            return features
            
        except Exception as e:
            logging.error(f"Erro na extração de características de reconhecimento facial: {e}")
            return {
                'face_encodings_count': 0,
                'face_clusters_found': 0,
                'similar_faces_count': 0,
                'face_recognition_data': json.dumps({'error': str(e)})
            }


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
