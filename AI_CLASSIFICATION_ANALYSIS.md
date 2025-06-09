# Análise e Proposta de Sistema de Classificação com IA

## 📊 Análise dos Dados Atuais

### Dados Coletados no Sistema Atual

**✅ Pontos Fortes:**
- **Rótulos de Qualidade**: Escala 1-5 (⭐ a ⭐⭐⭐⭐⭐)
- **Categorias de Rejeição**: blur, dark, light, cropped, other
- **Metadados Temporais**: timestamps, session_id
- **Banco de Dados Estruturado**: SQLite com backup JSON

**📈 Distribuição Atual dos Dados:**
```
Total de Imagens Rotuladas: 23
├── Qualidade (15 imagens):
│   ├── Score 2: 1 imagem (6.7%)
│   ├── Score 4: 7 imagens (46.7%)
│   └── Score 5: 7 imagens (46.7%)
└── Rejeitadas (8 imagens):
    ├── Blur: 6 imagens (75%)
    ├── Cropped: 1 imagem (12.5%)
    └── Dark: 1 imagem (12.5%)
```

### 🔍 Análise dos Dados para IA

**✅ Dados Suficientes para Começar:**
1. **Classes bem definidas** (qualidade 1-5 + categorias de rejeição)
2. **Estrutura de dados consistente**
3. **Interface de rotulagem eficiente**

**❌ Limitações Identificadas:**
1. **Volume de dados insuficiente** (23 amostras vs. milhares necessárias)
2. **Desbalanceamento de classes** (ausência de scores 1 e 3)
3. **Falta de features automáticas** (metadados de imagem, features visuais)
4. **Ausência de validação cruzada**
5. **Falta de dados de contexto** (EXIF, dimensões, formato)

## 🎯 Proposta de Sistema de IA Robusto

### Fase 1: Coleta e Enriquecimento de Dados (2-4 semanas)

#### 1.1 Expansão do Dataset
- **Meta**: 1000+ imagens rotuladas por classe
- **Estratégia**: Rotulagem ativa com sugestões de IA
- **Prioridade**: Balanceamento das classes (especialmente scores 1 e 3)

#### 1.2 Extração Automática de Features
```python
Features Propostas:
├── Metadados EXIF
│   ├── Camera, ISO, Exposure, F-stop
│   ├── Dimensões, Formato, Tamanho do arquivo
│   └── Data/hora, GPS (se disponível)
├── Features Visuais
│   ├── Histogramas RGB/HSV
│   ├── Texturas (LBP, GLCM)
│   ├── Sharpness/Blur metrics
│   ├── Exposure/Brightness metrics
│   └── Composição (rule of thirds, symmetry)
├── Features Deep Learning
│   ├── Embeddings de CNNs pré-treinadas
│   ├── Features de objetos detectados
│   └── Features estéticas (NIMA, AVA)
└── Features de Contexto
    ├── Similaridade com outras imagens
    ├── Análise de faces/pessoas
    └── Análise de cenário (indoor/outdoor)
```

### Fase 2: Modelo de IA Multi-Modal (4-6 semanas)

#### 2.1 Arquitetura Proposta
```
Sistema Híbrido:
├── Módulo 1: Feature Engineering Clássico
│   ├── Random Forest para features numéricas
│   ├── SVM para features de textura
│   └── Ensemble de algoritmos tradicionais
├── Módulo 2: Deep Learning
│   ├── CNN pré-treinada (EfficientNet/ResNet)
│   ├── Transfer Learning com fine-tuning
│   └── Multi-task learning (qualidade + defeitos)
└── Módulo 3: Fusion Layer
    ├── Meta-learner para combinar predições
    ├── Confidence scoring
    └── Active learning para casos incertos
```

#### 2.2 Pipeline de Treinamento
1. **Pré-processamento**: Normalização, augmentation, balanceamento
2. **Feature Selection**: Análise de importância, PCA, correlação
3. **Model Selection**: Cross-validation, hyperparameter tuning
4. **Ensemble**: Stacking, voting, blending
5. **Validation**: Hold-out test set, métricas de negócio

### Fase 3: Sistema de Produção (2-3 semanas)

#### 3.1 API de Classificação
```python
# Endpoint de classificação automática
POST /api/classify
{
    "image_path": "path/to/image.jpg",
    "return_confidence": true,
    "return_explanations": true
}

Response:
{
    "predicted_quality": 4,
    "confidence": 0.87,
    "predicted_issues": ["slight_blur"],
    "explanations": {
        "quality_factors": {
            "sharpness": 0.8,
            "exposure": 0.9,
            "composition": 0.7
        }
    },
    "needs_human_review": false
}
```

#### 3.2 Sistema de Feedback Loop
- **Correções humanas** alimentam retreinamento
- **Monitoramento de drift** de dados
- **A/B testing** de modelos
- **Métricas de negócio** em tempo real

## 🛠️ Implementação Técnica

### Dados Adicionais Necessários

#### 1. Schema Expandido do Banco de Dados
```sql
-- Tabela de features automáticas
CREATE TABLE image_features (
    filename TEXT PRIMARY KEY,
    file_size INTEGER,
    width INTEGER,
    height INTEGER,
    format TEXT,
    
    -- EXIF data
    camera_make TEXT,
    camera_model TEXT,
    iso INTEGER,
    exposure_time REAL,
    f_number REAL,
    focal_length REAL,
    
    -- Computed features
    sharpness_score REAL,
    brightness_score REAL,
    contrast_score REAL,
    saturation_score REAL,
    noise_level REAL,
    
    -- Visual features (JSON)
    color_histogram TEXT,  -- JSON array
    texture_features TEXT, -- JSON object
    composition_score REAL,
    
    -- Deep learning embeddings
    cnn_features TEXT,     -- JSON array of embeddings
    
    extraction_timestamp TEXT
);

-- Tabela de predições do modelo
CREATE TABLE ai_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    model_version TEXT,
    predicted_quality INTEGER,
    predicted_issues TEXT, -- JSON array
    confidence_score REAL,
    prediction_timestamp TEXT,
    
    -- Feedback loop
    human_correction TEXT,
    correction_timestamp TEXT,
    was_correct BOOLEAN
);

-- Tabela de performance do modelo
CREATE TABLE model_metrics (
    model_version TEXT,
    metric_name TEXT,
    metric_value REAL,
    evaluation_date TEXT,
    dataset_size INTEGER
);
```

#### 2. Pipeline de Feature Extraction
```python
class AdvancedFeatureExtractor:
    def extract_all_features(self, image_path):
        return {
            **self.extract_exif_features(image_path),
            **self.extract_visual_features(image_path),
            **self.extract_deep_features(image_path),
            **self.extract_composition_features(image_path)
        }
    
    def extract_visual_features(self, image_path):
        """Extrai features visuais tradicionais"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        return {
            'sharpness': self.calculate_sharpness(gray),
            'brightness': np.mean(gray),
            'contrast': np.std(gray),
            'noise_level': self.estimate_noise(gray),
            'color_histogram': self.extract_color_histogram(img),
            'texture_features': self.extract_texture_features(gray)
        }
```

### Métricas de Avaliação

#### 1. Métricas Técnicas
- **Accuracy**: Precisão geral do modelo
- **F1-Score**: Por classe (especialmente para classes minoritárias)
- **Cohen's Kappa**: Concordância com rótulos humanos
- **AUC-ROC**: Para classificação binária (boa/ruim)
- **Mean Absolute Error**: Para scores de qualidade

#### 2. Métricas de Negócio
- **Tempo de Rotulagem**: Redução vs. processo manual
- **Throughput**: Imagens processadas por hora
- **Human-in-the-loop**: Taxa de casos que precisam revisão humana
- **Cost Savings**: ROI do sistema automatizado

## 📈 Roadmap de Implementação

### Semana 1-2: Preparação
- [ ] Expandir schema do banco de dados
- [ ] Implementar feature extraction automática
- [ ] Configurar pipeline de coleta de dados
- [ ] Criar sistema de anotação ativa

### Semana 3-4: Coleta de Dados
- [ ] Rotular 1000+ imagens com sistema melhorado
- [ ] Extrair features de todas as imagens
- [ ] Validar qualidade dos dados
- [ ] Implementar data augmentation

### Semana 5-6: Desenvolvimento do Modelo
- [ ] Treinar modelos baseline (RF, SVM)
- [ ] Implementar CNN com transfer learning
- [ ] Criar ensemble de modelos
- [ ] Validação cruzada e tuning

### Semana 7-8: Sistema de Produção
- [ ] Implementar API de classificação
- [ ] Integrar com interface web existente
- [ ] Sistema de monitoramento e feedback
- [ ] Testes A/B com usuários

### Semana 9-10: Otimização
- [ ] Análise de performance em produção
- [ ] Ajustes baseados em feedback real
- [ ] Implementar retreinamento automático
- [ ] Documentação e deploy final

## 💡 Próximos Passos Imediatos

1. **Expandir o banco de dados** com features automáticas
2. **Implementar extração de metadados** EXIF e features visuais
3. **Criar sistema de sugestões** na interface web
4. **Coletar mais dados** com foco no balanceamento
5. **Desenvolver modelo baseline** com dados atuais + features

**Investimento estimado**: 10-12 semanas de desenvolvimento
**ROI esperado**: 70-80% de redução no tempo de rotulagem manual
**Accuracy target**: 85%+ para classificação de qualidade

---

*Este documento será atualizado conforme o progresso do projeto.*
