# 📸 Sistema de Seleção de Fotos - Prompt de Refinamento

## 🎯 Objetivo Principal

Criar um sistema inteligente de análise fotográfica que identifique e classifique fotos com base em critérios técnicos e de composição, seguindo uma abordagem hierárquica que prioriza a **pessoa dominante** na imagem.

## 🔍 Pipeline de Análise Proposto

### **Etapa 1: Análise de Exposição (Base Técnica)**

```markdown
IMPLEMENTAR sistema de análise de exposição que:
- Calcule histograma RGB e HSV para determinar distribuição de luminosidade
- Identifique fotos sub-expostas (muito escuras) usando threshold < 40 no canal Value (HSV)
- Identifique fotos super-expostas (muito claras) usando threshold > 220 no canal Value (HSV)
- Use método de Otsu para determinar thresholds adaptativos baseados na distribuição da imagem
- Classifique como: ADEQUADA, ESCURA, CLARA, EXTREMAMENTE_ESCURA, EXTREMAMENTE_CLARA
```

**Implementação Sugerida:**
```python
def analyze_exposure(image):
    """
    Analyze image exposure using HSV histogram and adaptive thresholding
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value_channel = hsv[:, :, 2]
    
    # Calculate histogram
    hist = cv2.calcHist([value_channel], [0], None, [256], [0, 256])
    
    # Otsu threshold for adaptive analysis
    threshold, _ = cv2.threshold(value_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Mean brightness
    mean_brightness = np.mean(value_channel)
    
    # Classification
    if mean_brightness < 40:
        return ExposureLevel.EXTREMELY_DARK
    elif mean_brightness < 80:
        return ExposureLevel.DARK
    elif mean_brightness > 220:
        return ExposureLevel.EXTREMELY_BRIGHT
    elif mean_brightness > 180:
        return ExposureLevel.BRIGHT
    else:
        return ExposureLevel.ADEQUATE
```

### **Etapa 2: Detecção e Análise de Pessoas**

```markdown
IMPLEMENTAR detector de pessoas e rostos que:
- Use OpenCV Haar Cascades para detecção inicial de rostos (método já implementado)
- Integre YOLO v8 ou MediaPipe para detecção de pessoas completas
- Implemente análise de pose usando MediaPipe Pose para determinar posicionamento corporal
- Calcule área ocupada por cada pessoa detectada (bounding box)
- Determine pessoa dominante baseada em: área_relativa × posição_central × nitidez_local
```

**Implementação Sugerida:**
```python
def detect_persons_and_faces(image):
    """
    Detect persons and faces using multiple methods
    """
    # Face detection (already implemented)
    faces = detect_faces_haar(image)
    
    # Person detection using MediaPipe or YOLO
    persons = detect_persons_mediapipe(image)
    
    # Pose analysis
    poses = analyze_poses_mediapipe(image)
    
    return {
        'faces': faces,
        'persons': persons,
        'poses': poses
    }

def calculate_person_dominance(person_bbox, image_shape, local_sharpness):
    """
    Calculate dominance score for a detected person
    """
    x, y, w, h = person_bbox
    img_h, img_w = image_shape[:2]
    
    # Area ratio
    area_ratio = (w * h) / (img_w * img_h)
    
    # Centrality (distance from center)
    center_x, center_y = img_w // 2, img_h // 2
    person_center_x, person_center_y = x + w//2, y + h//2
    centrality = 1 - (np.sqrt((person_center_x - center_x)**2 + (person_center_y - center_y)**2) / 
                     np.sqrt(center_x**2 + center_y**2))
    
    # Combined dominance score
    dominance_score = (area_ratio * 0.4 + centrality * 0.3 + local_sharpness * 0.3)
    
    return dominance_score
```

### **Etapa 3: Identificação da Pessoa Dominante**

```markdown
IMPLEMENTAR algoritmo de ranqueamento que:
- Calcule score de dominância = (área_pessoa / área_total) × 0.4 + (centralidade_xy) × 0.3 + (nitidez_local) × 0.3
- Considere regra dos terços para determinar posicionamento ideal
- Avalie proximidade ao centro de interesse da imagem
- Identifique pessoa com maior score como "pessoa dominante"
- Extraia região de interesse (ROI) ampliada ao redor da pessoa dominante
```

**Implementação Sugerida:**
```python
def identify_dominant_person(persons_data, image):
    """
    Identify the dominant person in the image based on multiple criteria
    """
    if not persons_data:
        return None
    
    dominant_person = None
    max_score = 0
    
    for person in persons_data:
        # Calculate local sharpness in person ROI
        roi = extract_roi(image, person['bbox'], expand_factor=1.2)
        local_sharpness = calculate_variance_of_laplacian(roi)
        
        # Calculate dominance score
        dominance_score = calculate_person_dominance(
            person['bbox'], 
            image.shape, 
            local_sharpness
        )
        
        # Rule of thirds bonus
        thirds_bonus = calculate_rule_of_thirds_score(person['bbox'], image.shape)
        dominance_score += thirds_bonus * 0.1
        
        if dominance_score > max_score:
            max_score = dominance_score
            dominant_person = {
                **person,
                'dominance_score': dominance_score,
                'local_sharpness': local_sharpness,
                'roi': roi
            }
    
    return dominant_person
```

### **Etapa 4: Análise Específica da Pessoa Dominante**

```markdown
IMPLEMENTAR análise detalhada da pessoa dominante:
- Detecção de cortes: verifique se bounding box da pessoa toca as bordas da imagem
- Análise de blur local: calcule Variance of Laplacian apenas na ROI da pessoa
- Detecção de oclusão: identifique se objetos cobrem partes importantes (rosto, corpo)
- Análise de pose: determine se pessoa está em posição natural ou problemática
- Score de qualidade pessoal = função(corte, blur_local, oclusão, pose)
```

**Implementação Sugerida:**
```python
def analyze_dominant_person_quality(dominant_person, image):
    """
    Detailed quality analysis of the dominant person
    """
    bbox = dominant_person['bbox']
    roi = dominant_person['roi']
    
    # Cropping detection
    cropping_issues = detect_person_cropping(bbox, image.shape)
    
    # Local blur analysis
    local_blur_score = calculate_variance_of_laplacian(roi)
    
    # Occlusion detection
    occlusion_level = detect_occlusion(roi, dominant_person.get('pose', {}))
    
    # Pose analysis
    pose_quality = analyze_pose_quality(dominant_person.get('pose', {}))
    
    # Combined quality score
    quality_score = calculate_person_quality_score(
        cropping_issues, local_blur_score, occlusion_level, pose_quality
    )
    
    return {
        'cropping_issues': cropping_issues,
        'local_blur_score': local_blur_score,
        'occlusion_level': occlusion_level,
        'pose_quality': pose_quality,
        'overall_quality': quality_score
    }

def detect_person_cropping(bbox, image_shape):
    """
    Detect if person is cropped at image boundaries
    """
    x, y, w, h = bbox
    img_h, img_w = image_shape[:2]
    
    cropping_issues = []
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
```

### **Etapa 5: Reconhecimento e Agrupamento Facial**

```markdown
IMPLEMENTAR sistema de identificação facial usando:
- Face Recognition library (baseada em dlib) para extração de encodings faciais
- Calcule distância euclidiana entre encodings para determinar similaridade
- Use threshold de 0.6 para determinar se duas faces são da mesma pessoa
- Implemente clustering DBSCAN para agrupar faces similares automaticamente
- Crie sistema de tags/IDs para cada cluster de pessoa identificada
- Permita filtragem e busca por pessoa específica
```

**Implementação Sugerida:**
```python
import face_recognition
from sklearn.cluster import DBSCAN

def extract_face_encodings(image, face_locations):
    """
    Extract face encodings for recognition
    """
    encodings = face_recognition.face_encodings(image, face_locations)
    return encodings

def cluster_faces(face_encodings, eps=0.6):
    """
    Cluster similar faces using DBSCAN
    """
    if not face_encodings:
        return []
    
    # DBSCAN clustering
    clustering = DBSCAN(eps=eps, metric='euclidean')
    cluster_labels = clustering.fit_predict(face_encodings)
    
    return cluster_labels

def create_person_database(images_with_faces):
    """
    Create database of persons from clustered faces
    """
    all_encodings = []
    all_metadata = []
    
    for image_path, faces_data in images_with_faces.items():
        for face in faces_data:
            all_encodings.append(face['encoding'])
            all_metadata.append({
                'image_path': image_path,
                'face_location': face['location'],
                'confidence': face.get('confidence', 1.0)
            })
    
    # Cluster all faces
    cluster_labels = cluster_faces(all_encodings)
    
    # Group by person
    persons_db = {}
    for i, label in enumerate(cluster_labels):
        if label == -1:  # Noise/outlier
            continue
            
        person_id = f"person_{label}"
        if person_id not in persons_db:
            persons_db[person_id] = []
        
        persons_db[person_id].append({
            'encoding': all_encodings[i],
            'metadata': all_metadata[i]
        })
    
    return persons_db
```

## 🛠️ Métodos Científicos Recomendados

### **Para Análise de Exposição:**
- **Histogram Equalization** (Gonzalez & Woods, 2018)
- **Adaptive Thresholding** (Otsu, 1979)
- **Zone System Analysis** (Adams, 1948) - adaptado para digital

### **Para Detecção de Pessoas:**
- **Haar Cascade Classifiers** (Viola & Jones, 2001) - já implementado
- **YOLO (You Only Look Once)** (Redmon et al., 2016) - para detecção robusta
- **MediaPipe** (Google, 2020) - para análise de pose e landmarks

### **Para Análise de Qualidade:**
- **Variance of Laplacian** (Pech-Pacheco et al., 2000) - já implementado
- **Sobel Gradient Magnitude** para detecção de bordas
- **Structural Similarity Index (SSIM)** para comparação de qualidade

### **Para Reconhecimento Facial:**
- **FaceNet** (Schroff et al., 2015) - via face_recognition library
- **Deep Face Recognition** (Parkhi et al., 2015)
- **DBSCAN Clustering** (Ester et al., 1996) para agrupamento automático

## 📊 Estrutura de Dados Proposta

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class PersonAnalysis:
    """Analysis data for a detected person"""
    person_id: int
    dominance_score: float
    bounding_box: Tuple[int, int, int, int]  # x, y, w, h
    pose_landmarks: Optional[List[Tuple[float, float]]]
    face_encoding: Optional[np.ndarray]
    quality_score: float
    cropping_issues: List[str]
    blur_score_local: float
    occlusion_level: float
    face_locations: List[Tuple[int, int, int, int]]

@dataclass
class PhotoQualityAnalysis:
    """Complete photo quality analysis"""
    filename: str
    exposure_classification: str  # ADEQUATE, DARK, BRIGHT, etc.
    persons_detected: List[PersonAnalysis]
    dominant_person: Optional[PersonAnalysis]
    technical_quality: float
    composition_score: float
    recommendation: str  # KEEP, REJECT, REVIEW
    
    # Additional metrics
    overall_blur_score: float
    brightness_score: float
    contrast_score: float
    color_harmony_score: float
    
    # Metadata
    analysis_timestamp: str
    analysis_version: str
```

## 🎛️ Configurações Recomendadas

```json
{
  "person_analysis": {
    "detection": {
      "min_person_area_ratio": 0.05,
      "face_detection_scale_factor": 1.1,
      "face_detection_min_neighbors": 4,
      "use_mediapipe": true,
      "use_yolo": false
    },
    "dominance_calculation": {
      "weights": {
        "area": 0.4,
        "centrality": 0.3,
        "local_sharpness": 0.3
      },
      "rule_of_thirds_bonus": 0.1
    },
    "face_recognition": {
      "encoding_model": "large",
      "similarity_threshold": 0.6,
      "clustering_eps": 0.5,
      "min_samples": 2
    }
  },
  "exposure_analysis": {
    "thresholds": {
      "extremely_dark": 40,
      "dark": 80,
      "bright": 180,
      "extremely_bright": 220
    },
    "use_adaptive_thresholds": true,
    "histogram_bins": 256
  },
  "quality_thresholds": {
    "blur_local_threshold": 50,
    "cropping_tolerance": 0.1,
    "occlusion_max_acceptable": 0.3,
    "minimum_face_size": 30
  },
  "analysis_settings": {
    "roi_expand_factor": 1.2,
    "pose_quality_enabled": true,
    "advanced_composition_analysis": true
  }
}
```

## 🔄 Integração com Sistema Existente

### **1. Extensão do FeatureExtractor**

```python
# Em src/core/feature_extractor.py
def _extract_person_features(self, image):
    """Extract person-specific features"""
    persons = detect_persons_and_faces(image)
    dominant_person = identify_dominant_person(persons, image)
    
    if dominant_person:
        person_quality = analyze_dominant_person_quality(dominant_person, image)
        face_encodings = extract_face_encodings(image, dominant_person['face_locations'])
        
        return {
            'dominant_person_score': dominant_person['dominance_score'],
            'dominant_person_quality': person_quality['overall_quality'],
            'dominant_person_cropped': len(person_quality['cropping_issues']) > 0,
            'dominant_person_blur': person_quality['local_blur_score'],
            'face_encodings': face_encodings,
            'total_persons': len(persons),
            'total_faces': len(dominant_person.get('face_locations', []))
        }
    
    return {
        'dominant_person_score': 0,
        'dominant_person_quality': 0,
        'dominant_person_cropped': False,
        'dominant_person_blur': 0,
        'face_encodings': [],
        'total_persons': 0,
        'total_faces': 0
    }
```

### **2. Atualização do Banco de Dados**

```sql
-- Adições à tabela image_features
ALTER TABLE image_features ADD COLUMN exposure_classification TEXT;
ALTER TABLE image_features ADD COLUMN dominant_person_score REAL;
ALTER TABLE image_features ADD COLUMN dominant_person_quality REAL;
ALTER TABLE image_features ADD COLUMN dominant_person_cropped BOOLEAN;
ALTER TABLE image_features ADD COLUMN dominant_person_blur REAL;
ALTER TABLE image_features ADD COLUMN total_persons INTEGER;
ALTER TABLE image_features ADD COLUMN face_encodings TEXT; -- JSON array

-- Nova tabela para reconhecimento facial
CREATE TABLE IF NOT EXISTS face_clusters (
    cluster_id TEXT PRIMARY KEY,
    representative_encoding TEXT NOT NULL, -- JSON array
    image_count INTEGER DEFAULT 0,
    created_timestamp TEXT NOT NULL,
    updated_timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS face_instances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    cluster_id TEXT,
    face_encoding TEXT NOT NULL, -- JSON array
    face_location TEXT NOT NULL, -- JSON array [top, right, bottom, left]
    confidence REAL DEFAULT 1.0,
    is_dominant_person BOOLEAN DEFAULT 0,
    FOREIGN KEY (cluster_id) REFERENCES face_clusters(cluster_id)
);
```

### **3. Expansão da Interface Web**

```python
# Em src/web/app.py - adicionar novas categorias de rejeição
rejection_keys = {
    'd': {'reason': 'dark', 'label': '🌑 Muito Escura'},
    'l': {'reason': 'light', 'label': '☀️ Muito Clara'},
    'b': {'reason': 'blur', 'label': '😵‍💫 Muito Borrada'},
    'c': {'reason': 'cropped', 'label': '✂️ Cortada/Incompleta'},
    'p': {'reason': 'person_cropped', 'label': '👤 Pessoa Cortada'},
    'f': {'reason': 'person_blurred', 'label': '👤😵‍💫 Pessoa Borrada'},
    'o': {'reason': 'occlusion', 'label': '🚫 Pessoa Obstruída'},
    'x': {'reason': 'other', 'label': '❌ Outros Problemas'}
}

# Nova rota para filtrar por pessoa
@self.app.route('/api/person/<cluster_id>')
def get_person_images(cluster_id):
    """Get all images containing a specific person"""
    # Implementation here
    pass
```

## 📈 Métricas de Avaliação

```python
evaluation_metrics = {
    "person_detection_accuracy": "% de pessoas corretamente detectadas",
    "dominant_person_precision": "% de pessoas dominantes corretamente identificadas", 
    "face_recognition_accuracy": "% de faces corretamente agrupadas",
    "cropping_detection_recall": "% de cortes detectados",
    "exposure_classification_accuracy": "% de classificações de exposição corretas",
    "overall_quality_correlation": "Correlação com avaliação manual",
    "processing_speed": "Imagens processadas por segundo"
}

def evaluate_person_analysis_system(test_dataset):
    """
    Evaluate the person analysis system against ground truth
    """
    results = {
        'person_detection': {'tp': 0, 'fp': 0, 'fn': 0},
        'dominant_person': {'correct': 0, 'total': 0},
        'face_clustering': {'accuracy': 0, 'silhouette_score': 0},
        'cropping_detection': {'precision': 0, 'recall': 0},
        'processing_time': []
    }
    
    for test_image in test_dataset:
        start_time = time.time()
        
        # Run analysis
        analysis = analyze_photo_complete(test_image['image'])
        
        # Evaluate results
        evaluate_person_detection(analysis, test_image['ground_truth'], results)
        evaluate_dominant_person(analysis, test_image['ground_truth'], results)
        
        processing_time = time.time() - start_time
        results['processing_time'].append(processing_time)
    
    return calculate_final_metrics(results)
```

## 🚀 Plano de Implementação

### **Fase 1: Análise de Exposição e Detecção Básica (Semana 1-2)**
- [ ] Implementar análise de exposição com histogramas HSV
- [ ] Integrar detecção de pessoas com MediaPipe
- [ ] Testar com dataset de 100 imagens
- [ ] Validar precisão da detecção de exposição

### **Fase 2: Pessoa Dominante e Análise Específica (Semana 3-4)**
- [ ] Implementar algoritmo de ranqueamento de dominância
- [ ] Desenvolver análise de qualidade específica da pessoa
- [ ] Criar detecção de cortes e oclusão
- [ ] Testar com dataset expandido (500 imagens)

### **Fase 3: Reconhecimento Facial (Semana 5-6)**
- [ ] Integrar face_recognition library
- [ ] Implementar clustering DBSCAN para faces
- [ ] Criar banco de dados de pessoas
- [ ] Desenvolver interface de busca por pessoa

### **Fase 4: Interface e Usabilidade (Semana 7)**
- [ ] Expandir interface web com novos filtros
- [ ] Adicionar visualização de análise de pessoas
- [ ] Implementar funcionalidade de agrupamento por pessoa
- [ ] Testes de usabilidade com usuários

### **Fase 5: Otimização e Deploy (Semana 8)**
- [ ] Otimizar performance do processamento
- [ ] Implementar processamento em batch
- [ ] Criar documentação completa
- [ ] Deploy em ambiente de produção

## 📚 Dependências Adicionais

```bash
# Instalação das dependências necessárias
pip install face-recognition
pip install mediapipe
pip install ultralytics  # Para YOLO se escolhido
pip install scikit-learn  # Para clustering DBSCAN
pip install scipy  # Para análises estatísticas avançadas
```

## 🎯 Resultados Esperados

### **Melhoria na Precisão de Classificação:**
- **Detecção de pessoas**: 95%+ de precisão
- **Identificação de pessoa dominante**: 90%+ de precisão
- **Reconhecimento facial**: 85%+ de precisão no agrupamento
- **Detecção de problemas específicos**: 80%+ de recall

### **Funcionalidades Novas:**
- Filtragem por pessoa específica
- Detecção automática de fotos com pessoas cortadas
- Análise de qualidade focada na pessoa principal
- Agrupamento automático por pessoa
- Recomendações inteligentes baseadas na análise da pessoa

### **Impacto no Workflow:**
- Redução de 60% no tempo de curadoria manual
- Melhoria de 40% na precisão de seleção automática
- Interface mais intuitiva para fotógrafos
- Maior confiabilidade nas decisões automáticas

---

*Documento criado em 23 de junho de 2025 - Sistema Photo Culling v2.0*
