# Advanced Face Detection
# Implementação robusta de detecção de rostos com múltiplos algoritmos

import cv2
import numpy as np
import os

class AdvancedFaceDetector:
    """
    Detector avançado de rostos usando múltiplos algoritmos e análise contextual
    para reduzir falsos positivos em imagens sem rostos.
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.debug = self.config.get('debug', False)
        
        # Inicializar detectores
        self.detectors = self._initialize_detectors()
        
    def _initialize_detectors(self):
        """Inicializa múltiplos detectores de rosto"""
        detectors = {}
        
        try:
            # 1. Haarcascade - Frontal
            frontal_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(frontal_cascade_path):
                detectors['frontal'] = cv2.CascadeClassifier(frontal_cascade_path)
            
            # 2. Haarcascade - Profile
            profile_cascade_path = cv2.data.haarcascades + 'haarcascade_profileface.xml'
            if os.path.exists(profile_cascade_path):
                detectors['profile'] = cv2.CascadeClassifier(profile_cascade_path)
            
            # 3. DNN Face Detector (OpenCV)
            try:
                # Tentar carregar modelo DNN se disponível
                dnn_config = self.config.get('dnn_face_detection', {})
                if dnn_config.get('enabled', False):
                    model_path = dnn_config.get('model_path')
                    config_path = dnn_config.get('config_path')
                    
                    if model_path and config_path and os.path.exists(model_path):
                        detectors['dnn'] = cv2.dnn.readNetFromTensorflow(model_path, config_path)
            except Exception as e:
                if self.debug:
                    print(f"DNN face detector não disponível: {e}")
        
        except Exception as e:
            if self.debug:
                print(f"Erro ao inicializar detectores: {e}")
        
        return detectors
    
    def detect_image_context(self, image_path):
        """
        Analisa o contexto da imagem para determinar se deve conter rostos
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"expected_faces": False, "context": "unknown", "confidence": 0}
            
            height, width = image.shape[:2]
            aspect_ratio = width / height
            image_size = height * width
            
            # Análise de composição e conteúdo
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            context_scores = {}
            
            # 1. Análise de proporção
            if aspect_ratio > 2.0 or aspect_ratio < 0.5:
                context_scores['landscape_panorama'] = 0.8
            elif 0.7 <= aspect_ratio <= 1.4:
                context_scores['portrait_square'] = 0.7
            
            # 2. Análise de complexidade de textura
            texture_complexity = self._analyze_texture_complexity(gray)
            context_scores['texture_complexity'] = texture_complexity
            
            # 3. Análise de cores (detecção de pele)
            skin_ratio = self._detect_skin_presence(image)
            context_scores['skin_presence'] = skin_ratio
            
            # 4. Análise de padrões geométricos (arquitetura, objetos)
            geometric_score = self._detect_geometric_patterns(gray)
            context_scores['geometric_patterns'] = geometric_score
            
            # 5. Análise de gradientes (céu, água, paisagens)
            gradient_uniformity = self._analyze_gradient_uniformity(gray)
            context_scores['gradient_uniformity'] = gradient_uniformity
            
            # Classificação de contexto
            context, expected_faces, confidence = self._classify_context(context_scores, aspect_ratio)
            
            return {
                "expected_faces": expected_faces,
                "context": context,
                "confidence": confidence,
                "scores": context_scores
            }
            
        except Exception as e:
            if self.debug:
                print(f"Erro na análise de contexto: {e}")
            return {"expected_faces": True, "context": "unknown", "confidence": 0}
    
    def _analyze_texture_complexity(self, gray_image):
        """Analisa complexidade de textura (alta = objetos/pessoas, baixa = céu/água)"""
        # Usar Local Binary Pattern simplificado
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        
        filtered = cv2.filter2D(gray_image, -1, kernel)
        texture_variance = np.var(filtered)
        
        # Normalizar baseado no tamanho da imagem
        normalized_variance = texture_variance / (gray_image.shape[0] * gray_image.shape[1])
        
        return min(1.0, normalized_variance / 1000)  # Escalar para 0-1
    
    def _detect_skin_presence(self, image):
        """Detecta presença de tons de pele na imagem"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Múltiplos ranges para diferentes tons de pele
        skin_ranges = [
            ([0, 20, 70], [20, 255, 255]),    # Tom claro
            ([0, 30, 80], [25, 255, 255]),    # Tom médio
            ([0, 40, 60], [30, 255, 200])     # Tom escuro
        ]
        
        total_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in skin_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            total_mask = cv2.bitwise_or(total_mask, mask)
        
        # Filtrar ruído usando operações morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        total_mask = cv2.morphologyEx(total_mask, cv2.MORPH_CLOSE, kernel)
        total_mask = cv2.morphologyEx(total_mask, cv2.MORPH_OPEN, kernel)
        
        skin_pixels = cv2.countNonZero(total_mask)
        total_pixels = image.shape[0] * image.shape[1]
        
        return skin_pixels / total_pixels
    
    def _detect_geometric_patterns(self, gray_image):
        """Detecta padrões geométricos (arquitetura, objetos manufaturados)"""
        # Detectar linhas usando Hough Transform
        edges = cv2.Canny(gray_image, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        line_score = 0
        if lines is not None:
            line_score = min(1.0, len(lines) / 50)  # Normalizar
        
        # Detectar retângulos/contornos regulares
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regular_shapes = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filtrar contornos pequenos
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if 0.3 < circularity < 0.9:  # Formas regulares mas não círculos perfeitos
                        regular_shapes += 1
        
        shape_score = min(1.0, regular_shapes / 10)
        
        return (line_score + shape_score) / 2
    
    def _analyze_gradient_uniformity(self, gray_image):
        """Analisa uniformidade de gradientes (céu, água, fundos lisos)"""
        # Calcular gradientes
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calcular magnitude e direção
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Analisar uniformidade da magnitude
        magnitude_std = np.std(magnitude)
        magnitude_mean = np.mean(magnitude)
        
        if magnitude_mean > 0:
            uniformity = 1 - (magnitude_std / magnitude_mean)
            return max(0, uniformity)
        
        return 1.0  # Completamente uniforme
    
    def _classify_context(self, scores, aspect_ratio):
        """Classifica o contexto da imagem e prediz se deve conter rostos"""
        
        # Regras de classificação
        
        # Paisagem/Natureza
        if (scores.get('gradient_uniformity', 0) > 0.7 and 
            scores.get('texture_complexity', 0) < 0.3 and
            aspect_ratio > 1.3):
            return "landscape", False, 0.8
        
        # Arquitetura/Objetos
        if (scores.get('geometric_patterns', 0) > 0.6 and 
            scores.get('skin_presence', 0) < 0.05):
            return "architecture", False, 0.7
        
        # Retrato/Pessoas
        if (scores.get('skin_presence', 0) > 0.1 and 
            0.6 <= aspect_ratio <= 1.5):
            return "portrait", True, 0.8
        
        # Macro/Close-up
        if (scores.get('texture_complexity', 0) > 0.7 and 
            scores.get('geometric_patterns', 0) < 0.3):
            return "macro", False, 0.6
        
        # Evento/Grupo (horizontal com textura complexa)
        if (aspect_ratio > 1.5 and 
            scores.get('texture_complexity', 0) > 0.5 and
            scores.get('skin_presence', 0) > 0.05):
            return "group", True, 0.7
        
        # Default: indeterminado - assumir que pode ter rostos
        return "general", True, 0.4
    
    def robust_face_detection(self, image_path, context_info=None):
        """
        Detecção robusta de rostos usando múltiplos algoritmos e contexto
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"has_faces": False, "confidence": 0, "faces": [], "context": None}
            
            # Obter contexto se não fornecido
            if context_info is None:
                context_info = self.detect_image_context(image_path)
            
            # Se o contexto indica que não deveria ter rostos, aplicar threshold mais alto
            context_modifier = 1.0
            if not context_info["expected_faces"]:
                context_modifier = 1.5  # Threshold mais rigoroso
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_area = gray.shape[0] * gray.shape[1]
            
            all_faces = []
            detection_scores = {}
            
            # 1. Detecção Haarcascade Frontal
            if 'frontal' in self.detectors:
                frontal_faces = self._detect_frontal_faces(gray, context_modifier)
                all_faces.extend(frontal_faces)
                detection_scores['frontal'] = len(frontal_faces)
            
            # 2. Detecção Haarcascade Profile
            if 'profile' in self.detectors:
                profile_faces = self._detect_profile_faces(gray, context_modifier)
                all_faces.extend(profile_faces)
                detection_scores['profile'] = len(profile_faces)
            
            # 3. Detecção DNN (se disponível)
            if 'dnn' in self.detectors:
                dnn_faces = self._detect_dnn_faces(image, context_modifier)
                all_faces.extend(dnn_faces)
                detection_scores['dnn'] = len(dnn_faces)
            
            # Filtrar e consolidar detecções sobrepostas
            consolidated_faces = self._consolidate_detections(all_faces, image_area)
            
            # Análise de qualidade dos rostos detectados
            quality_faces = self._analyze_face_quality(gray, consolidated_faces, image_area)
            
            # Decisão final considerando contexto
            has_prominent_faces = self._make_final_decision(
                quality_faces, context_info, detection_scores
            )
            
            # Calcular confiança
            confidence = self._calculate_confidence(
                quality_faces, context_info, detection_scores
            )
            
            return {
                "has_faces": has_prominent_faces,
                "confidence": confidence,
                "faces": quality_faces,
                "context": context_info,
                "detection_scores": detection_scores
            }
            
        except Exception as e:
            if self.debug:
                print(f"Erro na detecção de rostos: {e}")
            return {"has_faces": True, "confidence": 0, "faces": [], "context": None}
    
    def _detect_frontal_faces(self, gray_image, context_modifier):
        """Detecta rostos frontais usando Haarcascade - MUITO MAIS PERMISSIVO"""
        if 'frontal' not in self.detectors:
            return []
        
        # Múltiplas tentativas com configurações diferentes
        all_faces = []
        
        # Tentativa 1: Configuração padrão mais permissiva
        faces1 = self.detectors['frontal'].detectMultiScale(
            gray_image,
            scaleFactor=1.05,  # Mais preciso
            minNeighbors=2,    # Muito menos restritivo
            minSize=(15, 15),  # Muito menor
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        all_faces.extend(faces1)
        
        # Tentativa 2: Ainda mais permissiva para rostos pequenos/distantes
        faces2 = self.detectors['frontal'].detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=1,    # Mínimo possível
            minSize=(10, 10),  # Ainda menor
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        all_faces.extend(faces2)
        
        # Remover duplicatas
        unique_faces = []
        for x, y, w, h in all_faces:
            is_duplicate = False
            for existing in unique_faces:
                if (abs(x - existing['x']) < 20 and abs(y - existing['y']) < 20):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_faces.append({"x": x, "y": y, "w": w, "h": h, "type": "frontal", "confidence": 0.8})
        
        return unique_faces
    
    def _detect_profile_faces(self, gray_image, context_modifier):
        """Detecta rostos de perfil usando Haarcascade - MUITO MAIS PERMISSIVO"""
        if 'profile' not in self.detectors:
            return []
        
        # Múltiplas tentativas com configurações diferentes
        all_faces = []
        
        # Tentativa 1: Configuração permissiva
        faces1 = self.detectors['profile'].detectMultiScale(
            gray_image,
            scaleFactor=1.05,
            minNeighbors=2,    # Menos restritivo
            minSize=(12, 12),  # Menor para perfis
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        all_faces.extend(faces1)
        
        # Tentativa 2: Muito permissiva
        faces2 = self.detectors['profile'].detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=1,    # Mínimo
            minSize=(8, 8),    # Ainda menor
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        all_faces.extend(faces2)
        
        # Remover duplicatas
        unique_faces = []
        for x, y, w, h in all_faces:
            is_duplicate = False
            for existing in unique_faces:
                if (abs(x - existing['x']) < 15 and abs(y - existing['y']) < 15):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_faces.append({"x": x, "y": y, "w": w, "h": h, "type": "profile", "confidence": 0.6})
        
        return unique_faces
    
    def _detect_dnn_faces(self, image, context_modifier):
        """Detecta rostos usando DNN (se disponível)"""
        if 'dnn' not in self.detectors:
            return []
        
        try:
            h, w = image.shape[:2]
            
            # Preparar input para DNN
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
            self.detectors['dnn'].setInput(blob)
            detections = self.detectors['dnn'].forward()
            
            faces = []
            confidence_threshold = 0.5 * context_modifier
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > confidence_threshold:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    
                    faces.append({
                        "x": x1, "y": y1, 
                        "w": x2 - x1, "h": y2 - y1, 
                        "type": "dnn", 
                        "confidence": float(confidence)
                    })
            
            return faces
            
        except Exception as e:
            if self.debug:
                print(f"Erro na detecção DNN: {e}")
            return []
    
    def _consolidate_detections(self, all_faces, image_area):
        """Consolida detecções sobrepostas de múltiplos detectores"""
        if not all_faces:
            return []
        
        # Agrupar faces sobrepostas
        consolidated = []
        used = set()
        
        for i, face1 in enumerate(all_faces):
            if i in used:
                continue
            
            group = [face1]
            used.add(i)
            
            for j, face2 in enumerate(all_faces[i+1:], i+1):
                if j in used:
                    continue
                
                # Calcular sobreposição
                overlap = self._calculate_overlap(face1, face2)
                
                if overlap > 0.3:  # 30% de sobreposição
                    group.append(face2)
                    used.add(j)
            
            # Consolidar grupo em uma única detecção
            best_face = max(group, key=lambda f: f['confidence'])
            consolidated.append(best_face)
        
        return consolidated
    
    def _calculate_overlap(self, face1, face2):
        """Calcula a sobreposição entre duas detecções de rosto"""
        x1_1, y1_1 = face1['x'], face1['y']
        x2_1, y2_1 = x1_1 + face1['w'], y1_1 + face1['h']
        
        x1_2, y1_2 = face2['x'], face2['y']
        x2_2, y2_2 = x1_2 + face2['w'], y1_2 + face2['h']
        
        # Calcular área de interseção
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calcular união
        area1 = face1['w'] * face1['h']
        area2 = face2['w'] * face2['h']
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _analyze_face_quality(self, gray_image, faces, image_area):
        """Analisa a qualidade e prominência dos rostos detectados"""
        quality_faces = []
        
        for face in faces:
            face_area = face['w'] * face['h']
            face_ratio = face_area / image_area
            
            # Análise de qualidade
            face_roi = gray_image[face['y']:face['y']+face['h'], 
                                  face['x']:face['x']+face['w']]
            
            quality_metrics = {
                'area_ratio': face_ratio,
                'size': face_area,
                'sharpness': np.var(cv2.Laplacian(face_roi, cv2.CV_64F)),
                'contrast': np.std(face_roi) / np.mean(face_roi) if np.mean(face_roi) > 0 else 0
            }
            
            # Score de qualidade combinado
            quality_score = (
                min(1.0, face_ratio * 50) * 0.4 +  # Tamanho relativo
                min(1.0, quality_metrics['sharpness'] / 100) * 0.3 +  # Nitidez
                min(1.0, quality_metrics['contrast']) * 0.2 +  # Contraste
                face['confidence'] * 0.1  # Confiança original
            )
            
            face_copy = face.copy()
            face_copy['quality_score'] = quality_score
            face_copy['quality_metrics'] = quality_metrics
            
            quality_faces.append(face_copy)
        
        # Ordenar por qualidade
        quality_faces.sort(key=lambda f: f['quality_score'], reverse=True)
        
        return quality_faces
    
    def _make_final_decision(self, quality_faces, context_info, detection_scores):
        """Toma decisão final sobre presença de rostos prominentes - MUITO MAIS PERMISSIVO"""
        
        # Se encontrou qualquer rosto, provavelmente tem pessoas
        if quality_faces:
            return True
        
        # Mesmo sem rostos de qualidade, verificar detecções brutas
        total_detections = sum(detection_scores.values())
        if total_detections > 0:
            return True  # Se qualquer detector encontrou algo, aceitar
        
        # Se contexto sugere possível presença de pessoas (cor de pele, etc)
        if context_info and context_info.get("scores", {}).get("skin_presence", 0) > 0.05:
            return True
        
        return False
    
    def _calculate_confidence(self, quality_faces, context_info, detection_scores):
        """Calcula confiança na decisão final"""
        if not quality_faces:
            base_confidence = 0.8 if not context_info["expected_faces"] else 0.3
        else:
            # Confiança baseada na melhor detecção
            best_face = quality_faces[0]
            base_confidence = best_face['quality_score']
        
        # Ajustar baseado no contexto
        context_confidence = context_info.get("confidence", 0.5)
        
        # Ajustar baseado na concordância entre detectores
        detector_agreement = len([score for score in detection_scores.values() if score > 0])
        agreement_bonus = min(0.2, detector_agreement * 0.1)
        
        final_confidence = min(1.0, base_confidence * context_confidence + agreement_bonus)
        
        return final_confidence
    
    def get_face_coordinates(self, image_path):
        """
        Função simplificada para obter coordenadas dos rostos detectados
        Retorna lista de retângulos (x, y, w, h) dos rostos encontrados
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = []
            
            # Usar detector frontal principal
            if 'frontal' in self.detectors:
                frontal_faces = self.detectors['frontal'].detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(30, 30)
                )
                
                for (x, y, w, h) in frontal_faces:
                    faces.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h), 'type': 'frontal'})
            
            # Se não encontrou rostos frontais, tentar perfil
            if len(faces) == 0 and 'profile' in self.detectors:
                profile_faces = self.detectors['profile'].detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(25, 25)
                )
                
                for (x, y, w, h) in profile_faces:
                    faces.append({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h), 'type': 'profile'})
            
            return faces
            
        except Exception as e:
            if self.debug:
                print(f"Erro ao obter coordenadas dos rostos: {e}")
            return []
