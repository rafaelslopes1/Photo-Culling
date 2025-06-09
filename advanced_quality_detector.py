# Advanced Image Quality Detection
# Implementação de algoritmos avançados para redução de falsos positivos

import cv2
import numpy as np
from scipy import ndimage
from scipy.fft import fft2, fftshift
import math
from PIL import Image, ExifTags

class AdvancedQualityDetector:
    """
    Classe para detecção avançada de qualidade de imagem com múltiplos algoritmos
    e thresholds adaptativos para reduzir falsos positivos.
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.debug = self.config.get('debug', False)
        
    def detect_image_type(self, image_path):
        """
        Detecta o tipo de imagem para aplicar critérios específicos
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return "unknown"
            
            height, width = image.shape[:2]
            aspect_ratio = width / height
            
            # Análise básica de conteúdo
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detectar se é paisagem (horizontal, pouco contraste vertical)
            if aspect_ratio > 1.5:
                # Análise de gradientes horizontais vs verticais
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                
                h_variance = np.var(grad_x)
                v_variance = np.var(grad_y)
                
                if h_variance > v_variance * 1.5:
                    return "landscape"
            
            # Detectar macro (alta variância local, baixa variância global)
            local_variance = self._calculate_local_variance(gray)
            global_variance = np.var(gray)
            
            if local_variance > global_variance * 2 and min(width, height) > 1000:
                return "macro"
            
            # Detectar retrato (proporção vertical, possível presença de pele)
            if aspect_ratio < 0.8:
                skin_ratio = self._detect_skin_ratio(image)
                if skin_ratio > 0.1:
                    return "portrait"
            
            return "general"
            
        except Exception as e:
            if self.debug:
                print(f"Erro na detecção de tipo: {e}")
            return "unknown"
    
    def _calculate_local_variance(self, gray_image, window_size=50):
        """Calcula variância local média"""
        h, w = gray_image.shape
        variances = []
        
        for y in range(0, h-window_size, window_size//2):
            for x in range(0, w-window_size, window_size//2):
                window = gray_image[y:y+window_size, x:x+window_size]
                variances.append(np.var(window))
        
        return np.mean(variances) if variances else 0
    
    def _detect_skin_ratio(self, image):
        """Detecta proporção aproximada de pixels de pele"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Definir range de cores de pele em HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Segunda faixa para tons de pele mais escuros
        lower_skin2 = np.array([0, 30, 80], dtype=np.uint8) 
        upper_skin2 = np.array([25, 255, 255], dtype=np.uint8)
        skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        
        combined_mask = cv2.bitwise_or(skin_mask, skin_mask2)
        
        total_pixels = image.shape[0] * image.shape[1]
        skin_pixels = cv2.countNonZero(combined_mask)
        
        return skin_pixels / total_pixels
    
    def advanced_blur_detection(self, image_path, image_type="general"):
        """
        Detecção avançada de blur usando múltiplos algoritmos
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"is_blurry": True, "confidence": 0, "scores": {}}
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            image_size = height * width
            
            scores = {}
            
            # 1. Variance of Laplacian (método original)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            scores['laplacian'] = laplacian_var
            
            # 2. Sobel Variance
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_variance = np.var(sobel_x) + np.var(sobel_y)
            scores['sobel'] = sobel_variance
            
            # 3. FFT-based blur detection
            fft_score = self._fft_blur_score(gray)
            scores['fft'] = fft_score
            
            # 4. Gradient Magnitude
            gradient_score = self._gradient_magnitude_score(gray)
            scores['gradient'] = gradient_score
            
            # 5. Edge Density
            edge_density = self._edge_density_score(gray)
            scores['edge_density'] = edge_density
            
            # Thresholds adaptativos baseados no tipo de imagem e tamanho
            thresholds = self._get_adaptive_blur_thresholds(image_type, image_size)
            
            # Análise multi-critério
            blur_indicators = 0
            confidence_scores = []
            
            for method, score in scores.items():
                threshold = thresholds.get(method, thresholds['default'])
                
                if score < threshold:
                    blur_indicators += 1
                    # Calcular confiança baseada na distância do threshold
                    confidence = max(0, (threshold - score) / threshold)
                    confidence_scores.append(confidence)
            
            # Decisão final baseada em consenso
            total_methods = len(scores)
            is_blurry = blur_indicators >= (total_methods * 0.6)  # 60% dos métodos devem concordar
            
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            
            # Ajustar confiança baseada no contexto
            if image_type == "macro" and not is_blurry:
                # Macro pode ter blur intencional (profundidade de campo)
                avg_confidence *= 0.7
            elif image_type == "landscape" and scores['gradient'] > thresholds['gradient'] * 0.7:
                # Paisagens podem ter regiões suaves intencionais
                avg_confidence *= 0.8
            
            return {
                "is_blurry": is_blurry,
                "confidence": avg_confidence,
                "scores": scores,
                "thresholds": thresholds,
                "blur_indicators": blur_indicators,
                "total_methods": total_methods
            }
            
        except Exception as e:
            if self.debug:
                print(f"Erro na detecção de blur: {e}")
            return {"is_blurry": True, "confidence": 0, "scores": {}}
    
    def _fft_blur_score(self, gray_image):
        """Score baseado na análise FFT (frequências altas)"""
        f_transform = fft2(gray_image)
        f_shift = fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        
        # Calcular energia em frequências altas
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Definir região de altas frequências (borda externa)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (center_w, center_h), min(center_h, center_w) // 3, 1, -1)
        mask = 1 - mask  # Inverter para pegar as bordas
        
        high_freq_energy = np.sum(magnitude_spectrum * mask)
        total_energy = np.sum(magnitude_spectrum)
        
        return (high_freq_energy / total_energy) * 1000000  # Escalar para comparação
    
    def _gradient_magnitude_score(self, gray_image):
        """Score baseado na magnitude do gradiente"""
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return np.mean(magnitude)
    
    def _edge_density_score(self, gray_image):
        """Score baseado na densidade de bordas"""
        edges = cv2.Canny(gray_image, 50, 150)
        edge_pixels = cv2.countNonZero(edges)
        total_pixels = gray_image.shape[0] * gray_image.shape[1]
        
        return (edge_pixels / total_pixels) * 100  # Percentual de pixels de borda
    
    def _get_adaptive_blur_thresholds(self, image_type, image_size):
        """Thresholds adaptativos baseados no tipo e tamanho da imagem"""
        base_thresholds = {
            'laplacian': 100,
            'sobel': 1000,
            'fft': 50,
            'gradient': 20,
            'edge_density': 5,
            'default': 100
        }
        
        # Ajustar baseado no tamanho da imagem
        size_multiplier = 1.0
        if image_size > 4000000:  # > 4MP
            size_multiplier = 1.3
        elif image_size < 1000000:  # < 1MP
            size_multiplier = 0.7
        
        # Ajustar baseado no tipo
        type_multipliers = {
            'landscape': 0.8,  # Paisagens podem ter regiões suaves
            'macro': 0.6,      # Macro pode ter blur intencional
            'portrait': 1.2,   # Retratos devem ser nítidos
            'general': 1.0
        }
        
        type_mult = type_multipliers.get(image_type, 1.0)
        final_multiplier = size_multiplier * type_mult
        
        return {k: v * final_multiplier for k, v in base_thresholds.items()}
    
    def advanced_lighting_analysis(self, image_path, image_type="general"):
        """
        Análise avançada de iluminação considerando contexto artístico
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"is_low_light": True, "confidence": 0, "analysis": {}}
            
            # Múltiplas análises de iluminação
            analysis = {}
            
            # 1. Análise básica de brilho (método original)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            avg_brightness = hsv[:, :, 2].mean()
            analysis['avg_brightness'] = avg_brightness
            
            # 2. Análise de histograma
            hist_analysis = self._analyze_brightness_histogram(hsv[:, :, 2])
            analysis.update(hist_analysis)
            
            # 3. Análise de contraste local
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            local_contrast = self._calculate_local_contrast(gray)
            analysis['local_contrast'] = local_contrast
            
            # 4. Detecção de clipping
            clipping_analysis = self._analyze_clipping(image)
            analysis.update(clipping_analysis)
            
            # 5. SNR estimation
            snr_score = self._estimate_snr(gray)
            analysis['snr'] = snr_score
            
            # 6. Detecção de low-key intencional
            is_lowkey_artistic = self._detect_lowkey_artistic(image, analysis)
            analysis['is_lowkey_artistic'] = is_lowkey_artistic
            
            # Decisão adaptativa
            thresholds = self._get_adaptive_lighting_thresholds(image_type)
            
            # Múltiplos critérios para decisão
            issues = []
            
            if avg_brightness < thresholds['brightness']:
                if not is_lowkey_artistic:
                    issues.append('low_brightness')
            
            if analysis['shadow_clipping'] > thresholds['shadow_clipping']:
                issues.append('shadow_clipping')
            
            if local_contrast < thresholds['local_contrast']:
                if not is_lowkey_artistic:
                    issues.append('low_contrast')
            
            if snr_score < thresholds['snr']:
                issues.append('high_noise')
            
            # Decisão final
            is_low_light = len(issues) >= 2  # Pelo menos 2 problemas
            
            # Se é low-key artístico, reduzir a confiança de problemas
            confidence = len(issues) / 4  # Máximo 4 problemas possíveis
            if is_lowkey_artistic and is_low_light:
                confidence *= 0.3  # Reduzir drasticamente se parece intencional
            
            return {
                "is_low_light": is_low_light,
                "confidence": confidence,
                "analysis": analysis,
                "issues": issues,
                "thresholds": thresholds
            }
            
        except Exception as e:
            if self.debug:
                print(f"Erro na análise de iluminação: {e}")
            return {"is_low_light": True, "confidence": 0, "analysis": {}}
    
    def _analyze_brightness_histogram(self, brightness_channel):
        """Análise detalhada do histograma de brilho"""
        hist = cv2.calcHist([brightness_channel], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        total_pixels = np.sum(hist)
        
        # Análise de distribuição
        shadows = np.sum(hist[:85]) / total_pixels  # 0-33%
        midtones = np.sum(hist[85:170]) / total_pixels  # 33-66%
        highlights = np.sum(hist[170:]) / total_pixels  # 66-100%
        
        # Peak analysis
        peak_position = np.argmax(hist)
        peak_concentration = hist[peak_position] / total_pixels
        
        return {
            'shadows_ratio': shadows,
            'midtones_ratio': midtones,
            'highlights_ratio': highlights,
            'peak_position': peak_position,
            'peak_concentration': peak_concentration
        }
    
    def _calculate_local_contrast(self, gray_image, window_size=50):
        """Calcula contraste local médio"""
        h, w = gray_image.shape
        contrasts = []
        
        for y in range(0, h-window_size, window_size//2):
            for x in range(0, w-window_size, window_size//2):
                window = gray_image[y:y+window_size, x:x+window_size]
                local_std = np.std(window)
                local_mean = np.mean(window)
                
                if local_mean > 0:
                    contrast = local_std / local_mean
                    contrasts.append(contrast)
        
        return np.mean(contrasts) if contrasts else 0
    
    def _analyze_clipping(self, image):
        """Análise de clipping em shadows e highlights"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        total_pixels = gray.shape[0] * gray.shape[1]
        
        # Shadow clipping (pixels completamente pretos)
        shadow_clipped = np.sum(gray <= 5)
        shadow_clipping_ratio = shadow_clipped / total_pixels
        
        # Highlight clipping (pixels completamente brancos)
        highlight_clipped = np.sum(gray >= 250)
        highlight_clipping_ratio = highlight_clipped / total_pixels
        
        return {
            'shadow_clipping': shadow_clipping_ratio,
            'highlight_clipping': highlight_clipping_ratio
        }
    
    def _estimate_snr(self, gray_image):
        """Estimativa simples de Signal-to-Noise Ratio"""
        # Usar filtro de suavização para estimar sinal
        signal = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Noise é a diferença entre original e sinal
        noise = gray_image.astype(np.float32) - signal.astype(np.float32)
        
        signal_power = np.var(signal)
        noise_power = np.var(noise)
        
        if noise_power > 0:
            snr = signal_power / noise_power
            return 10 * np.log10(snr)  # dB
        
        return 100  # SNR muito alto se noise_power ~ 0
    
    def _detect_lowkey_artistic(self, image, analysis):
        """Detecta se a imagem escura é intencional (low-key photography)"""
        # Critérios para low-key artístico:
        # 1. Boa distribuição tonal mesmo com baixo brilho
        # 2. Contraste adequado
        # 3. Sem clipping excessivo
        # 4. Peak no lado escuro mas com detalhes
        
        conditions = []
        
        # Boa distribuição mesmo sendo escura
        if analysis['midtones_ratio'] > 0.3 and analysis['highlights_ratio'] > 0.1:
            conditions.append(True)
        
        # Contraste local adequado
        if analysis['local_contrast'] > 0.3:
            conditions.append(True)
        
        # Sem clipping excessivo
        if analysis['shadow_clipping'] < 0.1:
            conditions.append(True)
        
        # Peak position na região escura mas não extrema
        if 30 < analysis['peak_position'] < 100:
            conditions.append(True)
        
        # Pelo menos 3 de 4 condições devem ser verdadeiras
        return sum(conditions) >= 3
    
    def _get_adaptive_lighting_thresholds(self, image_type):
        """Thresholds adaptativos para análise de iluminação"""
        base_thresholds = {
            'brightness': 50,
            'shadow_clipping': 0.15,
            'local_contrast': 0.25,
            'snr': 15
        }
        
        # Ajustar baseado no tipo
        if image_type == "portrait":
            base_thresholds['brightness'] = 60  # Retratos precisam de mais luz
            base_thresholds['local_contrast'] = 0.3
        elif image_type == "landscape":
            base_thresholds['brightness'] = 45  # Paisagens podem ser mais escuras
            base_thresholds['shadow_clipping'] = 0.2
        elif image_type == "macro":
            base_thresholds['local_contrast'] = 0.4  # Macro precisa de bom contraste
        
        return base_thresholds
