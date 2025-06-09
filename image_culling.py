#!/usr/bin/env python3
"""
Image Culling Pipeline - Enhanced Version with Advanced Quality Detection

Usage:
    python image_culling.py <input_folder> <output_folder> [--nsfw_model path/to/model.h5]
         [--blur_threshold 100] [--brightness_threshold 50] [--nsfw_threshold 0.7]

This script processes images and organizes them into categorized folders for easy manual review:

Input Processing:
  - Reads images from the input folder
  - Detects and categorizes duplicates (using perceptual hashing)
  - Advanced quality filters with multiple algorithms to reduce false positives
  - Contextual analysis for better classification accuracy
  - Optionally filters NSFW content (if model provided)
  - Calculates quality scores for approved images

Output Organization:
  📁 output/
    ├── selected/     - High-quality images ranked by score (001_85.23_IMG_0001.JPG)
    ├── duplicates/   - Duplicate images detected
    ├── blurry/       - Images that are too blurry
    ├── low_light/    - Images that are too dark
    ├── no_faces/     - Images without prominent faces (if enabled)
    ├── nsfw/         - NSFW content (if detection enabled)
    └── failed/       - Images that failed processing

This organization allows for easy manual review and recovery of images as needed.

Advanced Features:
  - Multiple blur detection algorithms (Laplacian, Sobel, FFT, Gradient)
  - Adaptive thresholds based on image type and size
  - Contextual lighting analysis (distinguishes artistic low-key from poor exposure)
  - Robust face detection with multiple algorithms and context awareness
  - Reduced false positives through intelligent classification
"""

import os
import cv2
import argparse
import shutil
import numpy as np
from PIL import Image
import imagehash
from datetime import datetime
import time
import json
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import threading

# Import advanced detectors
try:
    from advanced_quality_detector import AdvancedQualityDetector
    from advanced_face_detector import AdvancedFaceDetector
    ADVANCED_DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Detectores avançados não disponíveis: {e}")
    print("Usando métodos básicos de detecção.")
    ADVANCED_DETECTION_AVAILABLE = False

def load_config(config_path="config.json"):
    """
    Load configuration from JSON file if it exists, otherwise return default config.
    """
    default_config = {
        "processing_settings": {
            "blur_threshold": 25,  # Muito mais permissivo - apenas fotos muito borradas serão rejeitadas
            "brightness_threshold": 15,  # Muito mais permissivo - apenas fotos extremamente escuras serão rejeitadas
            "nsfw_threshold": 0.7,
            "quality_score_weights": {"sharpness": 1.0, "brightness": 1.0},
            "image_extensions": [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"],
            "progress_update_interval": 10,
            "multiprocessing": {
                "enabled": True,
                "max_workers": None,  # None = usar todos os cores disponíveis
                "chunk_size": 4  # Número de imagens por chunk
            },
            "face_detection": {
                "enabled": True,  # Ativando detecção de rostos
                "use_advanced_detection": True,  # Usar detecção avançada
                "min_face_size": 20,  # Muito menor para detectar rostos distantes
                "min_face_ratio": 0.003,  # Muito mais permissivo (0.3% vs 2%)
                "scale_factor": 1.05,  # Mais preciso
                "min_neighbors": 2,  # Menos restritivo
                "multiple_scales": True,  # Tentar múltiplas escalas
                "detect_partial_faces": True,  # Detectar rostos parciais
                "debug": False  # Para debugging quando necessário
            }
        },
        "output_folders": {
            "selected": "selected",
            "duplicates": "duplicates", 
            "blurry": "blurry",
            "low_light": "low_light",
            "nsfw": "nsfw",
            "no_faces": "no_faces",
            "failed": "failed"
        },
        "duplicate_detection": {
            "hash_size": 8,
            "hash_method": "average_hash"
        }
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                # Merge with defaults
                for section, values in loaded_config.items():
                    if section in default_config:
                        default_config[section].update(values)
                print(f"⚙️  Configurações carregadas de: {config_path}")
                return default_config
        except Exception as e:
            print(f"⚠️ Erro ao carregar configuração: {e}. Usando configuração padrão.")
    
    return default_config


def get_supported_extensions(config=None):
    """
    Get list of supported image extensions from config or use defaults.
    """
    if config and 'processing_settings' in config:
        return tuple(config['processing_settings'].get('image_extensions', 
                    ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp']))
    return ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')


# Try to import the NSFW detection module.
# If not installed or no model provided, we simply skip NSFW filtering.
try:
    from nsfw_detector import predict
except ImportError:
    predict = None
    print("nsfw_detector module not found. NSFW detection will be disabled.")


def find_duplicates(image_folder, config=None):
    """
    Find duplicate images in the folder using perceptual hashing.
    Returns a set of duplicate image file paths.
    """
    hash_dict = {}
    duplicates = set()
    processed_count = 0
    
    # Get supported extensions from config
    supported_extensions = get_supported_extensions(config)
    
    print("🔍 Iniciando detecção de duplicatas...")
    
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        # Only process files with supported image extensions
        if not image_path.lower().endswith(supported_extensions):
            continue
        try:
            img = Image.open(image_path)
            # You can use average_hash, phash, dhash, or whash.
            img_hash = str(imagehash.average_hash(img))
            processed_count += 1
            
            if img_hash in hash_dict:
                print(f"  💫 Duplicata encontrada: {image_name} é similar a {os.path.basename(hash_dict[img_hash])}")
                duplicates.add(image_path)
            else:
                hash_dict[img_hash] = image_path
        except Exception as e:
            print(f"  ❌ Erro ao processar {image_name} para detecção de duplicatas: {e}")
    
    print(f"✅ Detecção de duplicatas concluída. {processed_count} imagens analisadas, {len(duplicates)} duplicatas encontradas.\n")
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


def is_blurry(image_path, blur_threshold=100, config=None):
    """
    Determine if the image is blurry using advanced detection if available.
    """
    if ADVANCED_DETECTION_AVAILABLE and config and config['processing_settings'].get('advanced_quality_detection', {}).get('enabled', False):
        try:
            detector = AdvancedQualityDetector(config['processing_settings']['advanced_quality_detection'])
            image_type = detector.detect_image_type(image_path)
            result = detector.advanced_blur_detection(image_path, image_type)
            
            if config['processing_settings']['advanced_quality_detection'].get('debug', False):
                print(f"    🔍 Análise avançada de blur: {result}")
            
            return result['is_blurry']
        except Exception as e:
            print(f"    ⚠️ Erro na detecção avançada de blur, usando método básico: {e}")
    
    # Fallback para método básico
    score = variance_of_laplacian(image_path)
    return score < blur_threshold


def is_low_light(image_path, brightness_threshold=50, config=None):
    """
    Determine if the image is too dark using advanced analysis if available.
    """
    if ADVANCED_DETECTION_AVAILABLE and config and config['processing_settings'].get('advanced_quality_detection', {}).get('enabled', False):
        try:
            detector = AdvancedQualityDetector(config['processing_settings']['advanced_quality_detection'])
            image_type = detector.detect_image_type(image_path)
            result = detector.advanced_lighting_analysis(image_path, image_type)
            
            if config['processing_settings']['advanced_quality_detection'].get('debug', False):
                print(f"    💡 Análise avançada de iluminação: {result}")
            
            return result['is_low_light']
        except Exception as e:
            print(f"    ⚠️ Erro na análise avançada de iluminação, usando método básico: {e}")
    
    # Fallback para método básico
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


def has_prominent_faces(image_path, config=None):
    """
    Detect if the image has prominent faces using advanced detection if available.
    Uses more precise logic to avoid false positives.
    """
    if ADVANCED_DETECTION_AVAILABLE and config and config['processing_settings'].get('face_detection', {}).get('use_advanced_detection', False):
        try:
            detector = AdvancedFaceDetector(config['processing_settings'])
            
            # Primeiro detectar contexto da imagem
            context_info = detector.detect_image_context(image_path)
            
            # Usar detecção robusta
            result = detector.robust_face_detection(image_path, context_info)
            
            if config['processing_settings'].get('face_detection', {}).get('debug', False):
                print(f"    👤 Análise avançada de rostos: {result}")
            
            return result['has_faces']
        except Exception as e:
            print(f"    ⚠️ Erro na detecção avançada de rostos, usando método básico: {e}")
    
    # Fallback para método básico melhorado com lógica mais precisa
    face_settings = {
        "min_face_size": 20,
        "min_face_ratio": 0.003,
        "scale_factor": 1.05,
        "min_neighbors": 2
    }
    
    # Override with config settings if provided
    if config and 'processing_settings' in config and 'face_detection' in config['processing_settings']:
        face_settings.update(config['processing_settings']['face_detection'])
    
    # Check if face detection is disabled
    if not face_settings.get('enabled', True):
        return True  # If disabled, consider all images as having faces
    
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            return False
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_height, image_width = gray.shape
        image_area = image_height * image_width
        
        # LÓGICA MAIS PRECISA E RESTRITIVA
        
        # Contador de evidências de rostos
        face_evidence_score = 0
        total_face_area = 0
        
        # 1. Detector frontal padrão - mais restritivo
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces_frontal = face_cascade.detectMultiScale(
            gray,
            scaleFactor=face_settings.get('scale_factor', 1.05),
            minNeighbors=face_settings.get('min_neighbors', 3),  # Mais restritivo
            minSize=(face_settings.get('min_face_size', 30), face_settings.get('min_face_size', 30))  # Ligeiramente maior
        )
        
        # Avaliar rostos frontais
        valid_frontal_faces = 0
        for (x, y, w, h) in faces_frontal:
            face_area = w * h
            face_ratio = face_area / image_area
            
            # Verificar se o rosto é grande o suficiente e não é muito pequeno
            if face_ratio >= face_settings.get('min_face_ratio', 0.005):  # Ligeiramente mais restritivo
                valid_frontal_faces += 1
                total_face_area += face_area
                face_evidence_score += 3  # Peso alto para rostos frontais
        
        # 2. Detector de perfil - apenas se não encontrou rostos frontais suficientes
        if valid_frontal_faces == 0:
            try:
                profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
                faces_profile = profile_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=3,  # Mais restritivo
                    minSize=(25, 25)  # Maior que antes
                )
                
                for (x, y, w, h) in faces_profile:
                    face_area = w * h
                    face_ratio = face_area / image_area
                    if face_ratio >= 0.004:  # Mais restritivo
                        total_face_area += face_area
                        face_evidence_score += 2  # Peso médio para perfis
            except:
                pass
        
        # 3. Detecção de olhos - apenas como evidência adicional
        try:
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,  # Mais restritivo
                minSize=(8, 8)
            )
            
            # Agrupar olhos em pares (dois olhos próximos = provavelmente um rosto)
            eye_pairs = 0
            for i, (x1, y1, w1, h1) in enumerate(eyes):
                for j, (x2, y2, w2, h2) in enumerate(eyes[i+1:], i+1):
                    # Verificar se os olhos estão na mesma altura (linha horizontal)
                    y_diff = abs(y1 - y2)
                    x_diff = abs(x1 - x2)
                    
                    # Dois olhos devem estar próximos verticalmente e separados horizontalmente
                    if y_diff < 30 and 30 < x_diff < 200:
                        eye_pairs += 1
                        break
            
            if eye_pairs >= 1:
                face_evidence_score += 1  # Peso baixo para olhos
        except:
            pass
        
        # 4. Análise de tom de pele - apenas como evidência adicional
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Range mais restritivo para tons de pele
            lower_skin = np.array([0, 30, 80], dtype=np.uint8)  # Mais restritivo
            upper_skin = np.array([15, 255, 255], dtype=np.uint8)  # Mais restritivo
            
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Operações morfológicas para limpar a máscara
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            
            skin_pixels = cv2.countNonZero(skin_mask)
            skin_ratio = skin_pixels / image_area
            
            # Apenas se há uma quantidade significativa de pele E outras evidências
            if skin_ratio > 0.08 and face_evidence_score > 0:  # Mais restritivo
                face_evidence_score += 1  # Peso baixo para pele
        except:
            pass
        
        # DECISÃO FINAL - Ser mais restritivo
        # Precisa de pelo menos 3 pontos de evidência OU pelo menos 1 rosto frontal válido
        has_faces = (face_evidence_score >= 3) or (valid_frontal_faces >= 1)
        
        # Log de debug se ativado
        if config and config['processing_settings'].get('face_detection', {}).get('debug', False):
            print(f"    👤 Debug {os.path.basename(image_path)}: score={face_evidence_score}, frontal={valid_frontal_faces}, decision={has_faces}")
        
        return has_faces
        
    except Exception as e:
        print(f"  ⚠️ Erro na detecção de rostos para {os.path.basename(image_path)}: {e}")
        return True  # In case of error, don't discard the image


def quality_score(image_path, config=None):
    """
    Compute a quality score by combining the sharpness (variance of Laplacian)
    and brightness of the image with configurable weights.
    """
    weights = {"sharpness": 1.0, "brightness": 1.0}
    if config and 'processing_settings' in config:
        weights.update(config['processing_settings'].get('quality_score_weights', {}))
    
    blur_score = variance_of_laplacian(image_path)
    image = cv2.imread(image_path)
    if image is None:
        return 0
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness_score = hsv[:, :, 2].mean()
    
    # Calculate weighted score
    total_score = (blur_score * weights["sharpness"]) + (brightness_score * weights["brightness"])
    return total_score


def show_banner_and_settings(input_folder, output_folder, blur_threshold, brightness_threshold, nsfw_threshold, nsfw_model, config=None):
    """
    Display a banner with current settings before processing starts.
    """
    print("🎯 " + "="*68)
    print("🖼️  IMAGE CULLING PIPELINE - PROCESSAMENTO INTELIGENTE DE FOTOS")
    print("🎯 " + "="*68)
    print(f"📂 Pasta de entrada: {input_folder}")
    print(f"📁 Pasta de saída: {output_folder}")
    print(f"⚙️  Configurações:")
    print(f"   • Limite de nitidez (blur): {blur_threshold}")
    print(f"   • Limite de brilho: {brightness_threshold}")
    
    # Advanced detection status
    if ADVANCED_DETECTION_AVAILABLE and config and config['processing_settings'].get('advanced_quality_detection', {}).get('enabled', False):
        print(f"   🔬 Detecção avançada: ATIVADA")
        advanced_config = config['processing_settings']['advanced_quality_detection']
        if advanced_config.get('blur_detection', {}).get('use_multiple_algorithms', False):
            print(f"      • Múltiplos algoritmos de blur")
        if advanced_config.get('lighting_analysis', {}).get('detect_artistic_lowkey', False):
            print(f"      • Detecção de low-key artístico")
        if advanced_config.get('debug', False):
            print(f"      • Modo debug ativado")
    else:
        print(f"   🔧 Detecção básica: ATIVADA")
    
    if nsfw_model:
        print(f"   • Detecção NSFW: ATIVADA (limite: {nsfw_threshold})")
    else:
        print(f"   • Detecção NSFW: DESATIVADA")
    
    # Face detection settings
    face_enabled = True
    if config and 'processing_settings' in config and 'face_detection' in config['processing_settings']:
        face_settings = config['processing_settings']['face_detection']
        face_enabled = face_settings.get('enabled', True)
        if face_enabled:
            min_ratio = face_settings.get('min_face_ratio', 0.02) * 100
            if face_settings.get('use_advanced_detection', False) and ADVANCED_DETECTION_AVAILABLE:
                print(f"   👤 Detecção de rostos: AVANÇADA (min. {min_ratio:.1f}% da imagem)")
                print(f"      • Análise contextual ativada")
                print(f"      • Múltiplos algoritmos de detecção")
            else:
                print(f"   👤 Detecção de rostos: BÁSICA (min. {min_ratio:.1f}% da imagem)")
        else:
            print(f"   👤 Detecção de rostos: DESATIVADA")
    else:
        print(f"   👤 Detecção de rostos: BÁSICA (min. 2.0% da imagem)")
    
    # Multiprocessing settings
    if config and 'processing_settings' in config and 'multiprocessing' in config['processing_settings']:
        mp_settings = config['processing_settings']['multiprocessing']
        if mp_settings.get('enabled', True):
            max_workers = mp_settings.get('max_workers') or cpu_count()
            chunk_size = mp_settings.get('chunk_size', 4)
            print(f"   🚀 Processamento paralelo: ATIVADO ({max_workers} workers, chunks de {chunk_size})")
            print(f"      • Performance estimada: {max_workers}x mais rápido")
        else:
            print(f"   🐌 Processamento sequencial: ATIVADO")
    else:
        print(f"   🚀 Processamento paralelo: ATIVADO (padrão)")
    
    print("🎯 " + "="*68)
    print()


def process_single_image(args):
    """
    Worker function para processamento paralelo de uma única imagem.
    Retorna um dicionário com os resultados do processamento.
    """
    image_path, config, duplicates, nsfw_model = args
    
    result = {
        'image_path': image_path,
        'original_filename': os.path.basename(image_path),
        'category': None,
        'score': 0,
        'error': None
    }
    
    try:
        # Handle duplicates
        if image_path in duplicates:
            result['category'] = 'duplicates'
            return result

        # NSFW filtering (if model provided)
        if nsfw_model and is_nsfw(image_path, nsfw_model, config['processing_settings']['nsfw_threshold']):
            result['category'] = 'nsfw'
            return result

        # Check for blurriness
        if is_blurry(image_path, config['processing_settings']['blur_threshold'], config):
            result['category'] = 'blurry'
            return result

        # Check for low light
        if is_low_light(image_path, config['processing_settings']['brightness_threshold'], config):
            result['category'] = 'low_light'
            return result

        # Check for prominent faces (only if face detection is enabled)
        if config['processing_settings']['face_detection']['enabled'] and not has_prominent_faces(image_path, config):
            result['category'] = 'no_faces'
            return result

        # If the image passes all filters, calculate its quality score
        score = quality_score(image_path, config)
        result['category'] = 'selected'
        result['score'] = score
        return result

    except Exception as e:
        result['category'] = 'failed'
        result['error'] = str(e)
        return result


def process_images_parallel(input_folder, output_folder, config, duplicates, nsfw_model, folders):
    """
    Processa imagens em paralelo usando multiprocessing.
    """
    # Get supported extensions
    supported_extensions = get_supported_extensions(config)
    
    # Preparar lista de imagens para processar
    image_paths = []
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        if image_path.lower().endswith(supported_extensions):
            image_paths.append(image_path)
    
    total_images = len(image_paths)
    print(f"📊 Total de imagens para processar: {total_images}")
    
    # Configuração de multiprocessing
    mp_config = config['processing_settings'].get('multiprocessing', {})
    max_workers = mp_config.get('max_workers') or cpu_count()
    chunk_size = mp_config.get('chunk_size', 4)
    
    print(f"🚀 Processamento paralelo ativo: {max_workers} workers, chunks de {chunk_size}")
    print("-" * 60)
    
    # Preparar argumentos para os workers
    worker_args = [(path, config, duplicates, nsfw_model) for path in image_paths]
    
    # Inicializar estatísticas
    stats = {
        'total': 0,
        'selected': 0,
        'duplicates': len(duplicates),
        'blurry': 0,
        'low_light': 0,
        'nsfw': 0,
        'no_faces': 0,
        'failed': 0,
        'skipped': 0
    }
    
    ranked_images = []
    start_time = time.time()
    
    # Processar em paralelo
    with Pool(processes=max_workers) as pool:
        # Usar imap para ter feedback de progresso
        results = pool.imap(process_single_image, worker_args, chunksize=chunk_size)
        
        for i, result in enumerate(results, 1):
            # Atualizar progresso
            show_progress_bar(i, total_images, "Processando em paralelo", start_time=start_time)
            
            stats['total'] += 1
            category = result['category']
            original_filename = result['original_filename']
            
            # Contar estatísticas
            if category in stats:
                stats[category] += 1
            
            # Copiar arquivo para pasta de destino
            src_path = result['image_path']
            
            if category == 'selected':
                # Para imagens selecionadas, guardar para ranking posterior
                ranked_images.append((src_path, result['score']))
                print(f"  ✅ {original_filename} → selected/ (Score: {result['score']:.2f})")
            else:
                # Para outras categorias, copiar imediatamente
                dest_path = os.path.join(folders[category], original_filename)
                try:
                    shutil.copy(src_path, dest_path)
                    if category == 'duplicates':
                        print(f"  🔄 {original_filename} → duplicates/")
                    elif category == 'nsfw':
                        print(f"  🔞 {original_filename} → nsfw/")
                    elif category == 'blurry':
                        print(f"  💫 {original_filename} → blurry/")
                    elif category == 'low_light':
                        print(f"  🌑 {original_filename} → low_light/")
                    elif category == 'no_faces':
                        print(f"  👤 {original_filename} → no_faces/")
                    elif category == 'failed':
                        error_msg = result.get('error', 'Erro desconhecido')[:50]
                        print(f"  ❌ {original_filename} → failed/ ({error_msg}...)")
                except Exception as copy_error:
                    print(f"  💥 Erro ao copiar {original_filename}: {str(copy_error)[:30]}...")
    
    return ranked_images, stats


def process_images(input_folder, output_folder, nsfw_model=None,
                   blur_threshold=100, brightness_threshold=50, nsfw_threshold=0.7, config_path="config.json"):
    """
    Process all images in the input folder and organize them into categorized folders:
      - selected: Images that passed all filters (ranked by quality)
      - duplicates: Duplicate images
      - blurry: Images that are too blurry
      - low_light: Images that are too dark
      - nsfw: NSFW images (if NSFW detection is enabled)
      - failed: Images that could not be processed
    """
    start_time = time.time()
    
    # Load configuration
    config = load_config(config_path)
    
    # Override config with command line arguments if provided
    if blur_threshold != 100:  # Non-default value
        config['processing_settings']['blur_threshold'] = blur_threshold
    if brightness_threshold != 50:  # Non-default value
        config['processing_settings']['brightness_threshold'] = brightness_threshold
    if nsfw_threshold != 0.7:  # Non-default value
        config['processing_settings']['nsfw_threshold'] = nsfw_threshold
    
    # Show banner with settings
    show_banner_and_settings(input_folder, output_folder, 
                            config['processing_settings']['blur_threshold'], 
                            config['processing_settings']['brightness_threshold'], 
                            config['processing_settings']['nsfw_threshold'], nsfw_model, config)
    
    print("🚀 Iniciando processamento de imagens...")
    print("="*60)
    
    # Create organized output folders
    folders = {}
    for folder_name, folder_key in config['output_folders'].items():
        folders[folder_name] = os.path.join(output_folder, folder_key)
        os.makedirs(folders[folder_name], exist_ok=True)
    
    print("📁 Pastas de saída criadas:")
    for category, path in folders.items():
        print(f"  - {category}: {path}")
    print()

    print("🔄 Iniciando processamento das imagens...")
    duplicates = find_duplicates(input_folder, config)
    
    # Verificar se deve usar processamento paralelo
    use_multiprocessing = config['processing_settings'].get('multiprocessing', {}).get('enabled', True)
    
    if use_multiprocessing:
        # PROCESSAMENTO PARALELO
        ranked_images, stats = process_images_parallel(
            input_folder, output_folder, config, duplicates, nsfw_model, folders
        )
    else:
        # PROCESSAMENTO SEQUENCIAL (código original)
        # Get supported extensions
        supported_extensions = get_supported_extensions(config)
        
        # Count total images to process
        total_images = len([f for f in os.listdir(input_folder) 
                          if f.lower().endswith(supported_extensions)])
        print(f"📊 Total de imagens para processar: {total_images}")
        print("-" * 60)
        
        ranked_images = []
        stats = {
            'total': 0,
            'selected': 0,
            'duplicates': len(duplicates),
            'blurry': 0,
            'low_light': 0,
            'nsfw': 0,
            'no_faces': 0,
            'failed': 0,
            'skipped': 0
        }

        processed_counter = 0
        for image_name in os.listdir(input_folder):
            image_path = os.path.join(input_folder, image_name)
            # Only process image files
            if not image_path.lower().endswith(supported_extensions):
                stats['skipped'] += 1
                continue
            
            stats['total'] += 1
            processed_counter += 1
            original_filename = os.path.basename(image_path)
            
            # Show progress bar with ETA and current file
            show_progress_bar(processed_counter, total_images, f"Processando", start_time=start_time)
            print(f"  📸 {original_filename}")

            try:
                # Handle duplicates
                if image_path in duplicates:
                    dest_path = os.path.join(folders['duplicates'], original_filename)
                    shutil.copy(image_path, dest_path)
                    print(f"  🔄 Duplicata → duplicates/")
                    continue

                # NSFW filtering (if model provided)
                if nsfw_model and is_nsfw(image_path, nsfw_model, config['processing_settings']['nsfw_threshold']):
                    dest_path = os.path.join(folders['nsfw'], original_filename)
                    shutil.copy(image_path, dest_path)
                    stats['nsfw'] += 1
                    print(f"  🔞 NSFW → nsfw/")
                    continue

                # Check for blurriness
                if is_blurry(image_path, config['processing_settings']['blur_threshold'], config):
                    dest_path = os.path.join(folders['blurry'], original_filename)
                    shutil.copy(image_path, dest_path)
                    stats['blurry'] += 1
                    print(f"  💫 Borrada → blurry/")
                    continue

                # Check for low light
                if is_low_light(image_path, config['processing_settings']['brightness_threshold'], config):
                    dest_path = os.path.join(folders['low_light'], original_filename)
                    shutil.copy(image_path, dest_path)
                    stats['low_light'] += 1
                    print(f"  🌑 Escura → low_light/")
                    continue

                # Check for prominent faces (only if face detection is enabled)
                if config['processing_settings']['face_detection']['enabled'] and not has_prominent_faces(image_path, config):
                    dest_path = os.path.join(folders['no_faces'], original_filename)
                    shutil.copy(image_path, dest_path)
                    stats['no_faces'] += 1
                    print(f"  👤 Sem rostos em destaque → no_faces/")
                    continue

                # If the image passes all filters, calculate its quality score
                score = quality_score(image_path, config)
                ranked_images.append((image_path, score))
                stats['selected'] += 1
                print(f"  ✅ Aprovada → selected/ (Score: {score:.2f})")

            except Exception as e:
                # Handle images that failed to process
                dest_path = os.path.join(folders['failed'], original_filename)
                try:
                    shutil.copy(image_path, dest_path)
                    stats['failed'] += 1
                    print(f"  ❌ Erro → failed/ ({str(e)[:50]}...)")
                except Exception as copy_error:
                    print(f"  💥 Erro crítico: {original_filename} - {str(e)[:30]}... (Não foi possível copiar)")

    # Sort selected images by quality score in descending order (best first)
    ranked_images.sort(key=lambda x: x[1], reverse=True)

    # Copy selected images to the selected folder with ranking
    print(f"\n🏆 Organizando {len(ranked_images)} imagens selecionadas por qualidade...")
    for idx, (img_path, score) in enumerate(ranked_images):
        original_name = os.path.basename(img_path)
        name, ext = os.path.splitext(original_name)
        dest_filename = f"{idx + 1:03d}_{score:.2f}_{name}{ext}"
        dest_path = os.path.join(folders['selected'], dest_filename)
        
        try:
            shutil.copy(img_path, dest_path)
            if idx < 5:  # Show details for top 5 images
                print(f"  🥇 #{idx + 1}: {original_name} (Score: {score:.2f})")
            elif idx == 5 and len(ranked_images) > 5:
                print(f"  ... e mais {len(ranked_images) - 5} imagens organizadas")
        except Exception as e:
            print(f"  ❌ Erro ao copiar {original_name}: {e}")
            stats['failed'] += 1

    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Display final report
    display_final_report(stats, nsfw_model, processing_time)
    
    # Save processing summary to file
    save_processing_summary(output_folder, stats, nsfw_model, processing_time, ranked_images, config)


def display_final_report(stats, nsfw_model, processing_time):
    """
    Display the final processing report in the terminal.
    """
    print("\n" + "="*60)
    print("RELATÓRIO FINAL DE PROCESSAMENTO")
    print("="*60)
    print(f"⏱️  Tempo de processamento: {processing_time:.2f} segundos")
    print(f"📊 Total de arquivos processados: {stats['total']}")
    print(f"🚫 Arquivos não-imagem ignorados: {stats['skipped']}")
    print(f"")
    print(f"📸 Imagens SELECIONADAS: {stats['selected']}")
    print(f"🔄 Duplicatas encontradas: {stats['duplicates']}")
    print(f"💫 Imagens borradas: {stats['blurry']}")
    print(f"🌑 Imagens muito escuras: {stats['low_light']}")
    print(f"👤 Imagens sem rostos em destaque: {stats['no_faces']}")
    if nsfw_model:
        print(f"🔞 Imagens NSFW: {stats['nsfw']}")
    print(f"❌ Falhas no processamento: {stats['failed']}")
    print(f"")
    
    # Show processing rate
    if processing_time > 0:
        rate = stats['total'] / processing_time
        print(f"🚀 Taxa de processamento: {rate:.1f} imagens/segundo")
    
    print("✅ Todas as imagens foram organizadas nas pastas correspondentes para facilitar sua revisão manual!")
    print("="*60)


def save_processing_summary(output_folder, stats, nsfw_model, processing_time=None, ranked_images=None, config=None):
    """
    Save a detailed processing summary to a text file.
    """
    summary_path = os.path.join(output_folder, 'processing_summary.txt')
    
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("         RELATÓRIO DE PROCESSAMENTO DE IMAGENS\n")
            f.write("="*70 + "\n")
            f.write(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}\n")
            if processing_time:
                f.write(f"Tempo de processamento: {processing_time:.2f} segundos\n")
            f.write("\n")
            
            # Configuration used
            if config:
                f.write("⚙️  CONFIGURAÇÕES UTILIZADAS:\n")
                f.write("-" * 40 + "\n")
                settings = config.get('processing_settings', {})
                f.write(f"• Limite de nitidez (blur): {settings.get('blur_threshold', 'N/A')}\n")
                f.write(f"• Limite de brilho: {settings.get('brightness_threshold', 'N/A')}\n")
                f.write(f"• Limite NSFW: {settings.get('nsfw_threshold', 'N/A')}\n")
                
                face_settings = settings.get('face_detection', {})
                if face_settings.get('enabled', False):
                    f.write(f"• Detecção de rostos: Habilitada\n")
                    f.write(f"  - Tamanho mínimo do rosto: {face_settings.get('min_face_size', 'N/A')}% da imagem\n")
                    f.write(f"  - Fator de escala: {face_settings.get('scale_factor', 'N/A')}\n")
                    f.write(f"  - Vizinhos mínimos: {face_settings.get('min_neighbors', 'N/A')}\n")
                else:
                    f.write(f"• Detecção de rostos: Desabilitada\n")
                
                weights = settings.get('quality_score_weights', {})
                f.write(f"• Pesos de qualidade: nitidez={weights.get('sharpness', 1.0)}, brilho={weights.get('brightness', 1.0)}\n")
                extensions = settings.get('image_extensions', [])
                f.write(f"• Extensões suportadas: {', '.join(extensions)}\n")
                f.write("\n")
            
            # Estatísticas gerais
            f.write("📊 ESTATÍSTICAS GERAIS:\n")
            f.write("-" * 40 + "\n")
            total_processed = stats['total']
            f.write(f"• Total de arquivos analisados: {total_processed}\n")
            f.write(f"• Arquivos não-imagem ignorados: {stats['skipped']}\n")
            if total_processed > 0:
                success_rate = ((total_processed - stats['failed']) / total_processed) * 100
                f.write(f"• Taxa de sucesso: {success_rate:.1f}%\n")
            if processing_time and total_processed > 0:
                rate = total_processed / processing_time
                f.write(f"• Taxa de processamento: {rate:.1f} imagens/segundo\n")
            f.write("\n")
            
            # Categorização detalhada
            f.write("🗂️  CATEGORIZAÇÃO DETALHADA:\n")
            f.write("-" * 40 + "\n")
            categories = [
                ("📸 Imagens SELECIONADAS", stats['selected'], "verde"),
                ("🔄 Duplicatas detectadas", stats['duplicates'], "amarelo"),
                ("💫 Imagens borradas", stats['blurry'], "azul"),
                ("🌑 Imagens muito escuras", stats['low_light'], "roxo"),
                ("👤 Sem rostos em destaque", stats['no_faces'], "laranja"),
            ]
            
            if nsfw_model:
                categories.append(("🔞 Imagens NSFW", stats['nsfw'], "vermelho"))
            
            categories.append(("❌ Falhas no processamento", stats['failed'], "cinza"))
            
            for label, count, _ in categories:
                percentage = (count / total_processed * 100) if total_processed > 0 else 0
                f.write(f"{label}: {count:3d} ({percentage:5.1f}%)\n")
            
            f.write("\n")
            
            # Top imagens selecionadas
            if ranked_images and len(ranked_images) > 0:
                f.write("🏆 TOP 10 IMAGENS SELECIONADAS (por qualidade):\n")
                f.write("-" * 40 + "\n")
                top_images = sorted(ranked_images, key=lambda x: x[1], reverse=True)[:10]
                for i, (image_path, score) in enumerate(top_images, 1):
                    filename = os.path.basename(image_path)
                    f.write(f"{i:2d}. {filename} (Score: {score:.2f})\n")
                f.write("\n")
            
            # Estrutura de saída
            f.write("📁 ESTRUTURA DE SAÍDA:\n")
            f.write("-" * 40 + "\n")
            f.write("output/\n")
            f.write("├── selected/     → Imagens de alta qualidade (ranqueadas por score)\n")
            f.write("├── duplicates/   → Imagens duplicadas detectadas\n")
            f.write("├── blurry/       → Imagens com baixa nitidez\n")
            f.write("├── low_light/    → Imagens com baixa luminosidade\n")
            f.write("├── no_faces/     → Imagens sem rostos em destaque\n")
            if nsfw_model:
                f.write("├── nsfw/         → Conteúdo NSFW detectado\n")
            f.write("└── failed/       → Imagens com erro no processamento\n")
            f.write("\n")
            
            # Dicas para revisão manual
            f.write("💡 DICAS PARA REVISÃO MANUAL:\n")
            f.write("-" * 40 + "\n")
            f.write("• Verifique duplicates/ para imagens similares que podem ter valor\n")
            f.write("• Revise blurry/ para possíveis efeitos artísticos intencionais\n")
            f.write("• Analise low_light/ para fotos noturnas ou de ambiente\n")
            f.write("• Revise no_faces/ para paisagens, objetos ou retratos com rostos pequenos\n")
            f.write("• As imagens em selected/ estão ordenadas por qualidade (menor número = melhor)\n")
            f.write("• Use o arquivo config.json para ajustar thresholds para futuros processamentos\n")
            f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("Processamento concluído com sucesso! 🎉\n")
            f.write("="*70 + "\n")
            
        print(f"📄 Resumo detalhado salvo em: {summary_path}")
        
    except Exception as e:
        print(f"⚠️ Não foi possível salvar o resumo: {e}")


# Global variables for ETA calculation
_eta_history = []
_eta_window_size = 10

def show_progress_bar(current, total, prefix="Progresso", length=30, start_time=None):
    """
    Show a progress bar with estimated time remaining.
    Uses moving average for more stable ETA in multiprocessing environments.
    """
    global _eta_history
    
    if total == 0:
        return
    
    percent = current / total
    filled = int(length * percent)
    bar = "█" * filled + "░" * (length - filled)
    percentage = percent * 100
    
    # Calculate estimated time remaining with smoothing
    eta_text = ""
    if start_time and current > 0:
        elapsed = time.time() - start_time
        current_rate = current / elapsed
        
        if current_rate > 0:
            current_eta = (total - current) / current_rate
            
            # Add to history for smoothing
            _eta_history.append(current_eta)
            if len(_eta_history) > _eta_window_size:
                _eta_history.pop(0)
            
            # Use moving average for more stable ETA
            if len(_eta_history) >= 3:  # Need at least 3 samples
                # Remove outliers (values that are too different from median)
                sorted_history = sorted(_eta_history)
                median_eta = sorted_history[len(sorted_history)//2]
                
                # Filter out values that are more than 50% different from median
                filtered_history = [eta for eta in _eta_history 
                                  if abs(eta - median_eta) / median_eta < 0.5]
                
                if filtered_history:
                    smoothed_eta = sum(filtered_history) / len(filtered_history)
                else:
                    smoothed_eta = current_eta
                    
                if smoothed_eta > 60:
                    eta_text = f" ETA: {int(smoothed_eta//60)}m{int(smoothed_eta%60)}s"
                elif smoothed_eta > 0:
                    eta_text = f" ETA: {int(smoothed_eta)}s"
            else:
                # For first few samples, just show current rate
                eta_text = f" Taxa: {current_rate:.1f}/s"
    
    print(f"\r{prefix}: |{bar}| {current}/{total} ({percentage:.1f}%){eta_text}", end="", flush=True)
    
    if current == total:
        print()  # New line when complete
        _eta_history.clear()  # Reset for next run


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
    parser.add_argument("--config", default="config.json",
                        help="Path to configuration file (default: config.json).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: auto-detect).")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Disable multiprocessing (use sequential processing).")
    args = parser.parse_args()

    # Verify input folder exists
    if not os.path.exists(args.input_folder):
        print("❌ Erro: Pasta de entrada não existe!")
        exit(1)

    # Create output folder if it does not exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        print(f"📁 Pasta de saída criada: {args.output_folder}")

    # Load NSFW model if a path is provided and the module is available.
    nsfw_model = None
    if args.nsfw_model:
        if predict is None:
            print("⚠️ Aviso: Módulo nsfw_detector não instalado. Filtragem NSFW será ignorada.")
        else:
            try:
                nsfw_model = predict.load_model(args.nsfw_model)
                print(f"🔞 Modelo NSFW carregado de: {args.nsfw_model}")
            except Exception as e:
                print(f"❌ Falha ao carregar modelo NSFW: {e}")
                nsfw_model = None

    # Carregar config para aplicar configurações de multiprocessing
    config = load_config(args.config)
    
    # Aplicar configurações de linha de comando para multiprocessing
    if args.no_parallel:
        config['processing_settings']['multiprocessing']['enabled'] = False
        print("🐌 Processamento sequencial forçado via linha de comando")
    
    if args.workers is not None:
        config['processing_settings']['multiprocessing']['max_workers'] = args.workers
        print(f"👥 Número de workers definido: {args.workers}")
    
    # Salvar config temporariamente
    temp_config_path = args.config + ".tmp"
    with open(temp_config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    try:
        process_images(args.input_folder, args.output_folder, nsfw_model,
                       blur_threshold=args.blur_threshold,
                       brightness_threshold=args.brightness_threshold,
                       nsfw_threshold=args.nsfw_threshold,
                       config_path=temp_config_path)
    finally:
        # Limpar arquivo temporário
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


if __name__ == '__main__':
    # Proteção necessária para multiprocessing em alguns sistemas
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()

