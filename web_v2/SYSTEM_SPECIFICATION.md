# 📋 Photo Culling Web App v2.0 - Especificação Completa do Sistema

## 🎯 Visão Geral do Projeto

### Problema Central
O sistema atual de Photo Culling utiliza **thresholds fixos** calibrados manualmente, resultando em:
- ❌ Calibração demorada e imprecisa
- ❌ Dificuldade para diferentes tipos de fotografia
- ❌ Falta de adaptabilidade a novos cenários
- ❌ Desperdício de conhecimento especializado

### S    -- Categorical Assessments (baseadas em features técnicas)
    environment_lighting TEXT,  # muito_escuro, levemente_escuro, ideal, levemente_claro, muito_claro
    person_lighting TEXT,       # pessoa_muito_escura, pessoa_levemente_escura, ideal, pessoa_levemente_clara, pessoa_estourada
    person_sharpness_level TEXT, # muito_nitida, nitida, levemente_desfocada, moderadamente_desfocada, muito_desfocada
    person_position TEXT,       # centralizada, esquerda, direita, terco_superior, terco_inferior
    eyes_quality TEXT,          # muito_nitidos, nitidos, levemente_desfocados, desfocados, fechados_nao_visiveis
    technical_issues TEXT,      # JSON array: ["ruido_excessivo", "tremido", "foco_fundo", "pessoa_cortada"]Proposta
**Web App v2.0** transforma conhecimento fotográfico especializado em inteligência artificial através de:
- ✅ **Coleta estruturada** de avaliações de especialistas
- ✅ **Treinamento de modelos de IA** baseados em expertise real
- ✅ **Adaptabilidade automática** com mais dados
- ✅ **Precisão baseada em conhecimento profissional**

---

## 🎨 Experiência do Usuário (UX)

### Persona Principal: Fotógrafo Especialista
- **Perfil**: Fotógrafo profissional com 5+ anos de experiência
- **Objetivo**: Avaliar qualidade de fotos de forma rápida e precisa
- **Motivação**: Contribuir para melhoria do sistema de IA
- **Limitações**: Tempo limitado (máximo 30-45 minutos por sessão)

### Jornada do Usuário

#### 1. **Acesso Inicial** (30 segundos)
```
Login Simples → Identificação → Início da Sessão
```
- Interface limpa com campo único de identificação
- Sem complexidade de cadastro ou senhas
- Foco na rapidez para começar a avaliar

#### 2. **Avaliação de Imagens** (10-15 segundos por foto)
```
Visualização → Avaliação Multi-Dimensional → Decisões → Próxima
```
- **70% da tela**: Imagem em alta qualidade
- **30% da tela**: Painel de avaliação estruturada
- **Fluxo otimizado**: Minimizar cliques e movimentos

#### 3. **Feedback Contínuo** (Tempo real)
```
Progresso → Estatísticas → Tempo Médio → Resultados
```
- Barra de progresso visual
- Contadores de produtividade
- Feedback sobre velocidade de avaliação

---

## 🖥️ Interface do Usuário (UI)

### Layout Principal
```
┌─────────────────────────────────────────────────────────────┐
│  Progresso: [████████░░] 80% (240/300)    [← Prev] [Next →] │
├─────────────────────────────────┬───────────────────────────┤
│                                 │  🎯 Avaliação Especializada│
│                                 │                           │
│          IMAGEM                 │  📊 Info da Imagem        │
│         (70% width)             │  👥 Pessoas: 2  👁️ Faces: 1│
│                                 │  📐 3024x4032  📸 f/2.8   │
│                                 │                           │
│                                 │  ⭐ Qualidade Geral       │
│                                 │  [★★★★☆] 4/5              │
│                                 │                           │
│                                 │  🔍 Nitidez da Pessoa     │
│                                 │  [★★★☆☆] 3/5              │
│                                 │                           │
│                                 │  ✅ Aprovações            │
│                                 │  □ Portfólio   □ Cliente  │
│                                 │  □ Social      □ Edição   │
│                                 │                           │
│  [Zoom Fit] [Zoom 100%]        │  🎯 Confiança: 85%        │
│                                 │  [████████░] 85%          │
│                                 │                           │
│                                 │  💬 Comentários           │
│                                 │  [____________]           │
│                                 │                           │
│                                 │  [✅ Enviar] [❌ Rejeitar]│
└─────────────────────────────────┴───────────────────────────┘
```

### Componentes Detalhados

#### 1. **Painel de Imagem (70%)**
- **Fundo preto** para melhor contraste
- **Zoom inteligente**: Click para zoom, controles para ajustes
- **Navegação rápida**: Setas direcionais visíveis
- **Progresso visual**: Barra de progresso no topo
- **Metadados**: Nome do arquivo e contador

#### 2. **Painel de Avaliação (30%)**
- **Seções organizadas** por tipo de avaliação
- **Ratings visuais**: Sistema de estrelas interativo
- **Toggles modernos**: Switches para decisões binárias
- **Slider de confiança**: Indicação de certeza da avaliação
- **Campo de comentários**: Texto livre para observações

#### 3. **Sistema de Cores**
```css
Primary:   #2c3e50 (Azul escuro)
Secondary: #3498db (Azul)
Success:   #27ae60 (Verde)
Warning:   #f39c12 (Laranja)
Danger:    #e74c3c (Vermelho)
```

---

## ⚙️ Ferramentas e Funcionalidades

### 1. **Sistema de Avaliação Multi-Dimensional**

#### Ratings de Qualidade (1-5 estrelas)
- **Qualidade Geral**: Avaliação holística da imagem
- **Nitidez Geral**: Nitidez técnica da imagem toda
- **Nitidez da Pessoa**: Foco específico na pessoa principal
- **Qualidade da Exposição**: Iluminação e contraste
- **Qualidade da Composição**: Regras fotográficas e estética
- **Impacto Emocional**: Força emocional da imagem
- **Execução Técnica**: Aspectos técnicos (ISO, foco, etc.)

#### Avaliações Categóricas Específicas
**Exposição Geral da Imagem:**
- [ ] Muito Escuro (Subexposto)
- [ ] Levemente Escuro  
- [ ] Exposição Ideal
- [ ] Levemente Claro
- [ ] Muito Claro (Superexposto)

**Exposição da Pessoa Principal:**
- [ ] Pessoa Muito Escura
- [ ] Pessoa Levemente Escura
- [ ] Exposição Ideal na Pessoa
- [ ] Pessoa Levemente Clara
- [ ] Pessoa Estourada (Superexposta)

**Nitidez da Pessoa Principal:**
- [ ] Pessoa Muito Nítida (Foco Perfeito)
- [ ] Pessoa Nítida
- [ ] Levemente Desfocada
- [ ] Moderadamente Desfocada
- [ ] Muito Desfocada (Fora de Foco)

**Posição da Pessoa na Imagem:**
- [ ] Pessoa Centralizada
- [ ] Pessoa à Esquerda
- [ ] Pessoa à Direita
- [ ] Pessoa no Terço Superior
- [ ] Pessoa no Terço Inferior

**Qualidade dos Olhos (quando visíveis):**
- [ ] Olhos Muito Nítidos
- [ ] Olhos Nítidos
- [ ] Olhos Levemente Desfocados
- [ ] Olhos Desfocados
- [ ] Olhos Fechados/Não Visíveis

**Quantidade de Pessoas na Imagem:**
- [ ] Sem Pessoas Detectadas
- [ ] 1 Pessoa (Retrato Individual)
- [ ] 2 Pessoas (Casal/Dupla)
- [ ] 3-5 Pessoas (Grupo Pequeno)
- [ ] 6+ Pessoas (Grupo Grande)

**Contexto da Fotografia:**
- [ ] Ambiente Interno (Indoor)
- [ ] Ambiente Externo (Outdoor)
- [ ] Luz Natural (Dia)
- [ ] Luz Artificial (Flash/Estúdio)
- [ ] Contraluz/Backlight
- [ ] Horário Dourado (Golden Hour)

#### Decisões Binárias (Sim/Não)
- **Aprovar para Portfólio**: Qualidade para showcase profissional
- **Aprovar para Cliente**: Adequada para entrega ao cliente
- **Aprovar para Redes Sociais**: Boa para publicação online
- **Precisa de Edição**: Requer pós-processamento
- **Rejeição Completa**: Descarte definitivo

#### Metadados Contextuais
- **Nível de Confiança**: 0-100% (quão certo está o especialista)
- **Tempo de Avaliação**: Capturado automaticamente
- **Comentários**: Observações textuais livres
- **Issues Categorizados**: Problemas específicos identificados

### 2. **Atalhos de Teclado Otimizados**
```
1-5:        Rating rápido (qualidade geral)
Q:          Rejeição rápida
SPACE:      Próxima imagem
← →:        Navegação entre imagens
Z:          Zoom fit (ajustar à tela)
X:          Zoom 100%
Ctrl+Enter: Enviar avaliação
?:          Mostrar/ocultar ajuda
ESC:        Cancelar ação atual
```

### 3. **Funcionalidades de Produtividade**

#### Rejeição Rápida (Tecla Q)
- Define automaticamente: Qualidade = 1, Rejeição = True, Confiança = 90%
- Auto-submit em 0.5 segundos
- Para fotos obviamente inadequadas

#### Avaliação Express
- Rating 1-5 rápido com teclas numéricas
- Auto-preenche campos relacionados baseado no rating
- Acelera avaliação de casos óbvios

#### Navegação Inteligente
- **Memória de posição**: Lembra zoom e posição da imagem anterior
- **Pré-carregamento**: Carrega próxima imagem em background
- **Navegação fluida**: Transições suaves entre imagens

---

## 📱 Telas do Sistema

### 1. **Tela de Login** (`/login`)
```html
┌─────────────────────────────────────────┐
│  🎯 Photo Culling Expert System v2.0    │
│                                         │
│         Sistema de Avaliação            │
│         Especializada de Fotos          │
│                                         │
│  ┌─────────────────────────────────┐    │
│  │ ID do Especialista:             │    │
│  │ [photographer_expert_001      ] │    │
│  └─────────────────────────────────┘    │
│                                         │
│         [🚀 Iniciar Avaliação]          │
│                                         │
│  💡 Use um ID único para suas sessões   │
│                                         │
└─────────────────────────────────────────┘
```

### 2. **Tela Principal de Avaliação** (`/evaluate`)
- **Layout responsivo** 70/30
- **Interface otimizada** para velocidade
- **Feedback visual** contínuo
- **Atalhos visíveis** quando necessário

### 3. **Tela de Analytics** (`/analytics`)
```html
┌───────────────────────────────────────────────────────────┐
│  📊 Dashboard de Performance - Especialista: expert_001   │
├───────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐         │
│  │ 🖼️ Total    │ │ ⏱️ Tempo     │ │ 🎯 Precisão  │         │
│  │    847      │ │   12.3s     │ │    94%      │         │
│  │ Avaliadas   │ │ Por Imagem  │ │ Confiança   │         │
│  └─────────────┘ └─────────────┘ └─────────────┘         │
│                                                           │
│  📈 Distribuição de Ratings                              │
│  ★★★★★ [████████████████████] 45%                       │
│  ★★★★☆ [████████████] 30%                               │
│  ★★★☆☆ [██████] 15%                                     │
│  ★★☆☆☆ [███] 7%                                         │
│  ★☆☆☆☆ [█] 3%                                           │
│                                                           │
│  💼 Decisões de Aprovação                                │
│  Portfólio:     234 (28%) [████████]                     │
│  Cliente:       456 (54%) [█████████████]                │
│  Social:        612 (72%) [████████████████████]         │
│  Rejeições:     89 (11%)  [███]                          │
│                                                           │
│  [📥 Exportar Dados] [🔄 Nova Sessão]                    │
└───────────────────────────────────────────────────────────┘
```

### 4. **Tela de Conclusão** (`/completed`)
```html
┌─────────────────────────────────────────┐
│         🎉 Sessão Concluída!            │
│                                         │
│  📸 Imagens avaliadas: 250              │
│  ⏱️ Tempo total: 52 minutos             │
│  🚀 Velocidade média: 12.5s/imagem     │
│  🎯 Confiança média: 87%               │
│                                         │
│  🏆 Excelente trabalho!                 │
│  Suas avaliações ajudarão a treinar    │
│  uma IA mais precisa.                  │
│                                         │
│  [📊 Ver Analytics] [📥 Exportar]       │
│  [🔄 Nova Sessão]                       │
└─────────────────────────────────────────┘
```

---

## 🧠 Pipeline de Machine Learning

### 1. **Coleta de Dados**
```python
expert_evaluation = {
    "image_metadata": {
        "filename": "IMG_1234.jpg",
        "timestamp": "2025-06-25T18:30:00Z",
        "evaluator_id": "photographer_expert_001"
    },
    "ratings": {
        "overall_quality": 4,           # Qualidade geral (correlaciona com sharpness_laplacian, brightness_mean)
        "global_sharpness": 5,          # Nitidez geral da imagem (correlaciona com sharpness_laplacian)
        "person_sharpness": 3,          # Nitidez da pessoa (correlaciona com person_blur_score)
        "exposure_quality": 4,          # Qualidade de exposição (correlaciona com exposure_level, brightness_mean)
        "composition_quality": 5,       # Composição (correlaciona com rule_of_thirds_score, symmetry_score)
        "emotional_impact": 4,          # Impacto emocional (subjetivo, útil para ML)
        "technical_execution": 4        # Execução técnica (correlaciona com noise_level, contrast_rms)
    },
    "categorical_assessments": {
        "environment_lighting": "ideal",        # Correlaciona com brightness_mean, exposure_level
        "person_lighting": "ideal",             # Correlaciona com person_blur_score, overexposure features
        "person_sharpness_level": "nitida",     # Correlaciona com dominant_person_blur, person_blur_score
        "person_position": "centered",          # Correlaciona com dominant_person_bbox, centrality score
        "eyes_quality": "sharp",                # Correlaciona com face detection e face_blur_score
        "technical_issues": [                   # Correlaciona com noise_level, motion blur detection
            "ruido_excessivo"                   # Apenas problemas detectáveis tecnicamente
        ]
    },
    "decisions": {
        "approve_for_portfolio": true,
        "approve_for_client": true,
        "approve_for_social": true,
        "needs_editing": false,
        "complete_reject": false
    },
    "context": {
        "confidence_level": 0.87,
        "evaluation_time_seconds": 15,
        "comments": "Boa composição, pessoa um pouco desfocada mas adequada para uso"
    }
}
```

### 2. **Features Técnicas Combinadas**
```python
combined_features = {
    # Features técnicas automáticas (extraídas pelo FeatureExtractor)
    "technical": {
        "sharpness_laplacian": 156.7,           # Nitidez geral (Variance of Laplacian)
        "brightness_mean": 127.3,               # Brilho médio da imagem
        "contrast_rms": 45.2,                   # Contraste RMS
        "noise_level": 12.4,                    # Nível de ruído estimado
        "face_count": 1,                        # Número de faces detectadas
        "person_detection_confidence": 0.94,    # Confiança da detecção de pessoa
        "dominant_person_blur": 123.4,          # Blur específico da pessoa dominante
        "rule_of_thirds_score": 0.76,          # Score da regra dos terços
        "exposure_level": "adequate",           # Nível de exposição detectado
        "total_persons": 1,                     # Total de pessoas detectadas
        "dominant_person_score": 0.85,         # Score de dominância da pessoa
        "saturation_mean": 98.5                 # Saturação média da imagem
    },
    # Avaliações do especialista
    "expert": {
        "overall_quality": 4,
        "person_sharpness": 3,
        "confidence": 0.87
    }
}
```

### 3. **Modelos Treinados**
- **Quality Predictor**: Prediz ratings de qualidade (RMSE < 0.5)
- **Approval Predictor**: Prediz decisões de aprovação (Accuracy > 90%)
- **Confidence Estimator**: Estima confiança das predições
- **Issue Detector**: Identifica problemas específicos

### 4. **Active Learning Loop**
```python
def select_next_evaluation_batch(model, image_pool, batch_size=50):
    """
    Seleciona imagens mais informativas para próxima avaliação
    Prioriza casos onde modelo está incerto
    """
    uncertainties = model.predict_uncertainty(image_pool)
    diversity_scores = calculate_feature_diversity(image_pool)
    
    # Combina incerteza + diversidade
    selection_scores = 0.7 * uncertainties + 0.3 * diversity_scores
    
    return select_top_k(image_pool, selection_scores, batch_size)
```

---

## 🗂️ Estrutura de Dados

### Banco de Dados SQLite

#### Tabela: `expert_evaluations`
```sql
CREATE TABLE expert_evaluations (
    id INTEGER PRIMARY KEY,
    image_filename TEXT NOT NULL,
    evaluator_id TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    -- Ratings (1-5)
    overall_quality INTEGER,
    global_sharpness INTEGER,
    person_sharpness INTEGER,
    exposure_quality INTEGER,
    composition_quality INTEGER,
    emotional_impact INTEGER,
    technical_execution INTEGER,
    
    -- Categorical Assessments
    environment_lighting TEXT,  -- muito_escuro, levemente_escuro, ideal, levemente_claro, muito_claro
    person_lighting TEXT,       -- pessoa_muito_escura, pessoa_levemente_escura, ideal, pessoa_levemente_clara, pessoa_muito_clara
    person_sharpness_level TEXT, -- muito_nitida, nitida, levemente_desfocada, moderadamente_desfocada, muito_desfocada
    person_position TEXT,       -- centralizada, esquerda, direita, terco_superior, terco_inferior
    eyes_quality TEXT,          -- muito_nitidos, nitidos, levemente_desfocados, desfocados, fechados_nao_visiveis
    people_count TEXT,          -- sem_pessoas, 1_pessoa, 2_pessoas, 3_5_pessoas, 6_mais_pessoas
    photo_context TEXT,         -- interno, externo, luz_natural, luz_artificial, contraluz, golden_hour
    technical_issues TEXT,      -- JSON array: ["ruido_excessivo", "sombras_duras", etc.]
    
    -- Decisions (boolean)
    approve_for_portfolio BOOLEAN,
    approve_for_client BOOLEAN,
    approve_for_social BOOLEAN,
    needs_editing BOOLEAN,
    complete_reject BOOLEAN,
    
    -- Context
    confidence_level FLOAT,
    evaluation_time_seconds INTEGER,
    comments TEXT,
    issues TEXT -- JSON (legacy field, mantido para compatibilidade)
);
```

#### Tabela: `evaluation_sessions`
```sql
CREATE TABLE evaluation_sessions (
    id INTEGER PRIMARY KEY,
    evaluator_id TEXT NOT NULL,
    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_time DATETIME,
    images_evaluated INTEGER DEFAULT 0,
    total_time_seconds INTEGER DEFAULT 0,
    session_notes TEXT
);
```

### Arquivos de Export
```json
{
  "export_metadata": {
    "evaluator_id": "photographer_expert_001",
    "export_timestamp": "2025-06-25T20:30:00Z",
    "total_evaluations": 1247,
    "session_count": 8,
    "average_evaluation_time": 12.3
  },
  "evaluations": [
    {
      "image_filename": "IMG_1234.jpg",
      "ratings": { "overall_quality": 4, ... },
      "decisions": { "approve_for_portfolio": true, ... },
      "context": { "confidence_level": 0.87, ... }
    }
  ],
  "aggregate_stats": {
    "rating_averages": { "overall_quality": 3.8, ... },
    "approval_rates": { "portfolio": 0.28, "client": 0.54 },
    "confidence_distribution": [0.1, 0.2, 0.5, 0.15, 0.05]
  }
}
```

---

## 🛣️ Roadmap do Projeto

### **FASE 1: MVP Funcional** ⏳ (1-2 semanas)
*Status: Em desenvolvimento*

#### Entregáveis:
- [x] **Setup Inicial**
  - [x] Estrutura do projeto Flask
  - [x] Modelos de banco de dados
  - [x] Interface básica de login

- [x] **Interface Core**
  - [x] Layout responsivo 70/30
  - [x] Sistema de rating com estrelas
  - [x] Toggles para decisões binárias
  - [x] Navegação entre imagens

- [ ] **Funcionalidades Essenciais**
  - [x] Carregamento de imagens do data/input
  - [x] Salvamento de avaliações no SQLite
  - [ ] Sistema de progresso funcional
  - [ ] Atalhos de teclado básicos

#### Critério de Sucesso:
- ✅ Especialista consegue avaliar 10 imagens consecutivas
- ✅ Dados são salvos corretamente no banco
- ✅ Interface é responsiva e intuitiva

### **FASE 2: Otimização UX** 📱 (1-2 semanas)
*Status: Planejado*

#### Entregáveis:
- [ ] **Experiência Otimizada**
  - [ ] Todos os atalhos de teclado funcionais
  - [ ] Rejeição rápida (tecla Q)
  - [ ] Zoom e navegação fluida
  - [ ] Pré-carregamento de imagens

- [ ] **Dashboard Analytics**
  - [ ] Estatísticas de performance
  - [ ] Gráficos de distribuição
  - [ ] Exportação de dados

- [ ] **Validação com Usuário Real**
  - [ ] Sessão piloto com fotógrafo
  - [ ] Coleta de 500+ avaliações
  - [ ] Ajustes baseados em feedback

#### Critério de Sucesso:
- ⏱️ Tempo médio < 15 segundos por imagem
- 🎯 Taxa de conclusão de sessão > 90%
- 😊 Feedback positivo do especialista

### **FASE 3: Machine Learning** 🧠 (2-3 semanas)
*Status: Planejado*

#### Entregáveis:
- [ ] **Pipeline ML Básico**
  - [ ] Feature engineering combinado
  - [ ] Modelo de regressão para ratings
  - [ ] Modelo de classificação para aprovações
  - [ ] Validação cruzada

- [ ] **Active Learning**
  - [ ] Sistema de seleção inteligente
  - [ ] Métricas de incerteza
  - [ ] Loop de melhoria contínua

- [ ] **Integração com Sistema Principal**
  - [ ] API para predições
  - [ ] Migração de thresholds fixos
  - [ ] Comparação de performance

#### Critério de Sucesso:
- 🎯 Accuracy > 85% em decisões de aprovação
- 📊 RMSE < 0.5 em predições de rating
- 🚀 Performance superior aos thresholds fixos

### **FASE 4: Produção** 🚀 (2-4 semanas)
*Status: Planejado*

#### Entregáveis:
- [ ] **Sistema Robusto**
  - [ ] Handling de errors completo
  - [ ] Logs e monitoramento
  - [ ] Backup automático de dados
  - [ ] Testes automatizados

- [ ] **Múltiplos Especialistas**
  - [ ] Sistema de usuários
  - [ ] Agregação de avaliações
  - [ ] Detecção de outliers
  - [ ] Consenso entre especialistas

- [ ] **Deploy e Documentação**
  - [ ] Servidor de produção
  - [ ] Documentação completa
  - [ ] Manual do usuário
  - [ ] Plano de manutenção

#### Critério de Sucesso:
- 🔄 Sistema rodando 24/7 sem interrupções
- 👥 3+ especialistas utilizando regularmente
- 📈 Melhoria contínua demonstrável

### **FASE 5: Expansão** 🌟 (Futuro)
*Status: Visão*

#### Possíveis Entregáveis:
- [ ] **Especialização por Domínio**
  - [ ] Modelos específicos (retrato, paisagem, evento)
  - [ ] Transfer learning entre domínios
  - [ ] Expertise personalizada

- [ ] **Interface Avançada**
  - [ ] Versão mobile/tablet
  - [ ] Comparação side-by-side
  - [ ] Edição colaborativa

- [ ] **IA Avançada**
  - [ ] Computer Vision state-of-the-art
  - [ ] Modelos de atenção visual
  - [ ] Explicabilidade das decisões

---

## 📊 Métricas de Sucesso

### Métricas de Usuário
- **Velocidade**: < 15 segundos por avaliação
- **Engajamento**: > 90% taxa de conclusão de sessão
- **Satisfação**: Score > 4/5 em pesquisa pós-uso
- **Adoção**: 3+ especialistas utilizando regularmente

### Métricas Técnicas
- **Precisão ML**: > 85% accuracy em classificações
- **Performance**: < 0.5 RMSE em predições de rating
- **Confiabilidade**: 99.9% uptime do sistema
- **Escalabilidade**: Suporte a 10+ usuários simultâneos

### Métricas de Negócio
- **ROI**: 50% redução no tempo de calibração
- **Qualidade**: 30% melhoria na concordância com especialistas
- **Eficiência**: 70% menos intervenção manual necessária
- **Inovação**: Sistema de referência para outros projetos

---

## 🔧 Considerações Técnicas

### Performance
- **Frontend**: Otimização de carregamento de imagens
- **Backend**: Cache de features técnicas
- **Database**: Índices para queries frequentes
- **Network**: Compressão de imagens para web

### Segurança
- **Dados**: Backup automático das avaliações
- **Acesso**: Log de todas as sessões
- **Privacidade**: Anonimização opcional de dados
- **Integridade**: Validação de dados de entrada

### Manutenibilidade
- **Código**: Documentação inline completa
- **Testes**: Cobertura > 80%
- **Deploy**: Pipeline CI/CD
- **Monitoramento**: Logs estruturados e métricas

---

## 💡 Próximos Passos Imediatos

### 1. **Finalizar MVP** (Esta semana)
- [x] Interface principal funcionando
- [ ] Corrigir bugs de integração
- [ ] Testar fluxo completo end-to-end
- [ ] Ajustes finais de UX

### 2. **Sessão Piloto** (Próxima semana)
- [ ] Agendar com fotógrafo especialista
- [ ] Preparar 200-300 imagens diversas
- [ ] Coletar primeiras avaliações reais
- [ ] Iterar baseado no feedback

### 3. **ML Pipeline** (Semana seguinte)
- [ ] Implementar feature extraction combinado
- [ ] Treinar primeiros modelos
- [ ] Validar performance inicial
- [ ] Documentar resultados

---

**Este documento serve como norte para todo o desenvolvimento do sistema, garantindo que mantenhamos foco na experiência do usuário e nos objetivos de negócio.**

*Atualizado em: 25 de junho de 2025*

---

## 🔗 Correlação entre Avaliações e Features Técnicas

### Mapeamento Feature-Avaliação

O sistema foi otimizado para coletar apenas avaliações que se correlacionam diretamente com features técnicas extraíveis:

#### **Ratings de Qualidade (1-5 estrelas)**
- **Qualidade Geral** ↔ `sharpness_laplacian`, `brightness_mean`, `contrast_rms`
- **Nitidez Geral** ↔ `sharpness_laplacian`, `sharpness_sobel`, `sharpness_fft`  
- **Nitidez da Pessoa** ↔ `dominant_person_blur`, `person_blur_score`
- **Qualidade de Exposição** ↔ `exposure_level`, `brightness_mean`, `otsu_threshold`
- **Qualidade da Composição** ↔ `rule_of_thirds_score`, `symmetry_score`, `edge_density`
- **Execução Técnica** ↔ `noise_level`, `contrast_rms`, `saturation_mean`

#### **Avaliações Categóricas**
- **Exposição Geral** ↔ `brightness_mean`, `exposure_level`, `is_properly_exposed`
- **Exposição da Pessoa** ↔ `overexposure_features`, `person_lighting_analysis`
- **Nitidez da Pessoa** ↔ `dominant_person_blur`, `person_sharpness_level`, `face_blur_score`
- **Posição da Pessoa** ↔ `dominant_person_bbox`, `centrality`, `composition_score`
- **Qualidade dos Olhos** ↔ `face_count`, `face_detection_confidence`, `face_blur_score`
- **Problemas Técnicos** ↔ `noise_level`, `motion_blur_detection`, `cropping_issues`

#### **Decisões Binárias**
- **Aprovações** ↔ Combinação de múltiplas features técnicas via ML
- **Necessita Edição** ↔ `exposure_quality`, `noise_level`, `contrast_rms`
- **Rejeição Completa** ↔ Thresholds críticos de qualidade técnica

### Features Técnicas Disponíveis no Sistema

**Core Quality Features:**
- `sharpness_laplacian` - Variance of Laplacian (método principal de blur)
- `brightness_mean` - Brilho médio da imagem
- `contrast_rms` - Contraste RMS
- `noise_level` - Estimativa de ruído/granulação
- `saturation_mean` - Saturação média

**Person-Specific Features:**
- `total_persons` - Número de pessoas detectadas
- `dominant_person_score` - Score de dominância da pessoa principal
- `dominant_person_blur` - Blur específico da pessoa dominante
- `face_count` - Número de faces detectadas
- `dominant_person_cropped` - Se pessoa está cortada nas bordas

**Composition Features:**
- `rule_of_thirds_score` - Aplicação da regra dos terços
- `symmetry_score` - Score de simetria
- `edge_density` - Densidade de bordas
- `texture_complexity` - Complexidade de textura

**Exposure Features:**
- `exposure_level` - Nível de exposição (dark/adequate/bright)
- `exposure_quality_score` - Score numérico de qualidade de exposição
- `otsu_threshold` - Threshold automático de Otsu
- `is_properly_exposed` - Boolean de exposição adequada

### Benefícios da Correlação

1. **Treinamento de ML Efetivo**: Cada avaliação humana treina features específicas
2. **Validação de Features**: Avaliações confirmam se features técnicas estão corretas
3. **Otimização de Thresholds**: Especialistas ajudam a calibrar valores de corte
4. **Detecção de Outliers**: Discrepâncias entre técnico e humano indicam casos especiais
5. **Melhoria Contínua**: Sistema aprende com cada avaliação especializada

---
