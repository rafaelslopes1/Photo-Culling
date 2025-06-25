# 📊 Ferramentas de Análise e Qualidade - Photo Culling System v2.5

Este guia aborda as ferramentas especializadas de análise, qualidade e scores do sistema Photo Culling.

---

## 🎯 Visão Geral das Ferramentas

### 🔍 **quality_analyzer.py** - Análise de Qualidade Principal
Ferramenta principal para análise detalhada de qualidade de imagens com sistema de scores avançado.

### 📈 **analysis_tools.py** - Ferramentas Estatísticas 
Utilitários para análise estatística e métricas de performance do sistema.

### 📊 **visualization_tools.py** - Visualizações
Geração de gráficos, dashboards e visualizações dos dados de análise.

---

## 🔍 Sistema de Scores de Qualidade

### 📊 **Blur Score (Desfoque)**

#### Métricas Utilizadas
- **Variação do Laplaciano**: Método principal para detecção de blur
- **Gradient Magnitude**: Análise de bordas e definição
- **Fourier Transform**: Análise de frequência espacial

#### Categorização por Score
```
🟢 NÍTIDA       : Score > 78   (Excelente qualidade)
🟡 LEVEMENTE    : Score 50-78  (Boa qualidade, utilizável)
🟠 MODERADA     : Score 20-50  (Qualidade questionável)
🔴 BORRADA      : Score < 20   (Baixa qualidade, rejeitar)
```

#### Estratégias de Threshold
| Estratégia | Threshold | Taxa Rejeição | Uso Recomendado |
|------------|-----------|---------------|-----------------|
| `conservative` | 50 | ~30% | Arquivo pessoal/histórico |
| `balanced` | 78 | ~50% | **Uso geral (padrão)** |
| `aggressive` | 145 | ~70% | Portfólio profissional |
| `very_aggressive` | 98 | ~60% | Exposições/impressão |

### 📷 **Exposure Score (Exposição)**

#### Métricas de Exposição
- **Brightness Mean**: Brilho médio da imagem (0-255)
- **Histogram Analysis**: Distribuição de luminosidade
- **Dynamic Range**: Amplitude tonal da imagem

#### Categorização
```
🌞 SUPER_EXPOSTA  : Mean > 200  (Estourada, perda de detalhes)
🔆 SOBRE_EXPOSTA  : Mean 180-200 (Muito clara)
✅ BEM_EXPOSTA    : Mean 80-180  (Exposição adequada)
🌑 SUB_EXPOSTA    : Mean 40-80   (Escura mas recuperável)
⚫ MUITO_ESCURA   : Mean < 40    (Muito escura, difícil recuperar)
```

### 👤 **Person Score (Detecção de Pessoas)**

#### Métricas de Pessoa
- **Person Confidence**: Confiança da detecção (0-1)
- **Person Area Ratio**: Proporção da pessoa na imagem
- **Face Quality**: Qualidade da detecção facial
- **Pose Quality**: Qualidade da pose detectada

#### Scores Combinados
```
🏆 EXCELENTE     : Score > 0.8   (Pessoa clara e bem enquadrada)
✅ ÓTIMA         : Score 0.6-0.8 (Boa qualidade de pessoa)
🟡 MODERADA      : Score 0.4-0.6 (Pessoa detectada, qualidade média)
🟠 BAIXA         : Score 0.2-0.4 (Pessoa detectada, baixa qualidade)
❌ REJEITADA     : Score < 0.2   (Sem pessoa ou qualidade muito baixa)
```

---

## 🛠️ Como Usar as Ferramentas

### 📊 Análise Completa de Qualidade

```bash
# Análise completa com todos os scores
python tools/quality_analyzer.py --analyze

# Análise específica de blur
python tools/quality_analyzer.py --blur-only

# Análise com threshold personalizado
python tools/quality_analyzer.py --threshold 85

# Gerar relatório detalhado
python tools/quality_analyzer.py --detailed-report
```

### 📈 Análise Estatística

```bash
# Estatísticas gerais do projeto
python tools/analysis_tools.py --stats

# Análise de performance dos algoritmos
python tools/analysis_tools.py --performance

# Comparação de estratégias
python tools/analysis_tools.py --compare-strategies
```

### 📊 Visualizações

```bash
# Dashboard completo
python tools/visualization_tools.py --dashboard

# Gráficos de distribuição de scores
python tools/visualization_tools.py --score-distribution

# Histogramas de qualidade
python tools/visualization_tools.py --quality-histograms
```

---

## 📋 Interpretação dos Relatórios

### 🎯 **Relatório de Qualidade Típico**

```json
{
  "summary": {
    "total_analyzed": 150,
    "blur_analysis": {
      "sharp_images": 87,      // Score > 78
      "blurry_images": 63,     // Score < 78
      "blur_percentage": 42.0,
      "average_score": 76.3
    },
    "exposure_analysis": {
      "well_exposed": 120,     // 80-180 range
      "overexposed": 15,       // Mean > 180
      "underexposed": 15,      // Mean < 80
      "exposure_quality": 80.0
    },
    "person_analysis": {
      "with_persons": 95,      // Pessoas detectadas
      "high_quality": 67,      // Score > 0.6
      "person_quality": 70.5
    }
  }
}
```

### 📊 **Métricas de Performance**

```json
{
  "performance_metrics": {
    "processing_speed": {
      "images_per_second": 12.5,
      "average_time_per_image": 0.08
    },
    "accuracy_metrics": {
      "blur_detection_accuracy": 92.3,
      "person_detection_accuracy": 88.7,
      "overall_quality_score": 90.5
    },
    "efficiency_metrics": {
      "memory_usage_mb": 156.7,
      "cpu_usage_percent": 15.2,
      "gpu_utilization": 67.8
    }
  }
}
```

---

## 🎚️ Configuração Avançada

### ⚙️ **Personalização de Thresholds**

Edite `config.json` para ajustar parâmetros:

```json
{
  "quality_analysis": {
    "blur_detection": {
      "strategy": "balanced",
      "custom_threshold": 78,
      "enable_debug": false
    },
    "exposure_analysis": {
      "bright_threshold": 180,
      "dark_threshold": 80,
      "dynamic_range_min": 50
    },
    "person_analysis": {
      "confidence_threshold": 0.5,
      "min_person_area": 0.05,
      "face_detection_enabled": true
    }
  }
}
```

### 🔍 **Parâmetros de Análise**

| Parâmetro | Valor Padrão | Descrição |
|-----------|--------------|-----------|
| `blur_threshold` | 78 | Limite para classificar como nítida |
| `brightness_min` | 80 | Limite inferior de exposição adequada |
| `brightness_max` | 180 | Limite superior de exposição adequada |
| `person_confidence` | 0.5 | Confiança mínima para detectar pessoa |
| `face_min_size` | 0.02 | Tamanho mínimo da face (% da imagem) |

---

## 📈 Sugestões de Limpeza Baseadas em Scores

### 🎯 **Algoritmo de Recomendação**

O sistema gera sugestões automáticas baseadas nos scores:

1. **Imagens Extremamente Borradas** (Score < 20)
   - ✅ **Recomendação**: Remoção imediata
   - 💾 **Economia**: ~3MB por imagem
   - ⚠️ **Ação**: Automática (com confirmação)

2. **Imagens Muito Borradas** (Score 20-50)
   - 🔍 **Recomendação**: Revisão manual
   - 📋 **Critério**: Verificar valor sentimental/histórico
   - ⚠️ **Ação**: Manual

3. **Exposição Problemática**
   - 🌞 **Super-expostas**: Verificar recuperação possível
   - ⚫ **Muito escuras**: Avaliar processamento
   - 🎛️ **Ação**: Processamento adicional

4. **Sem Pessoas Detectadas** (quando esperado)
   - 👤 **Contexto**: Fotos de eventos/família
   - 🔍 **Ação**: Verificar falsos negativos
   - 📝 **Log**: Arquivar para revisão

### 🧹 **Scripts de Limpeza Automática**

```bash
# Gerar sugestões de limpeza
python tools/quality_analyzer.py --cleanup-suggestions

# Criar script de limpeza (não executa)
python tools/quality_analyzer.py --generate-cleanup-script

# Limpeza conservadora (apenas extremamente borradas)
python tools/quality_analyzer.py --conservative-cleanup

# Análise de economia de espaço
python tools/quality_analyzer.py --space-analysis
```

---

## 📊 Dashboard e Relatórios

### 🎨 **Visualizações Disponíveis**

1. **Distribution Charts**: Distribuição de scores de qualidade
2. **Quality Heatmaps**: Mapas de calor por diretório/data
3. **Performance Graphs**: Gráficos de performance temporal
4. **Comparison Charts**: Comparação entre estratégias
5. **Progress Tracking**: Acompanhamento de melhorias

### 📋 **Tipos de Relatório**

- **📊 Executive Summary**: Resumo executivo para tomada de decisão
- **🔍 Detailed Analysis**: Análise técnica detalhada
- **📈 Performance Report**: Métricas de performance e eficiência
- **🧹 Cleanup Report**: Relatório de limpeza e otimização
- **📅 Historical Trends**: Tendências históricas de qualidade

---

## 🚀 Automação e Monitoramento

### ⏰ **Análise Automatizada**

```bash
# Análise diária (cron job)
0 9 * * * cd /path/to/Photo-Culling && python tools/quality_analyzer.py --daily-report

# Análise semanal completa
0 9 * * 0 cd /path/to/Photo-Culling && python tools/quality_analyzer.py --weekly-analysis

# Monitoramento contínuo
python tools/quality_analyzer.py --monitor --interval 3600
```

### 📊 **Alertas Inteligentes**

O sistema gera alertas quando:
- 📈 **Aumento significativo** de imagens borradas (>10% em relação à média)
- 💾 **Uso excessivo de espaço** por imagens de baixa qualidade
- ⚡ **Degradação de performance** nos algoritmos
- 🎯 **Oportunidades de limpeza** (>100MB em imagens rejeitáveis)

---

## 🎓 Exemplos Práticos

### 📷 **Análise de Sessão Fotográfica**

```bash
# Analisar fotos de evento específico
python tools/quality_analyzer.py --dir "data/input/evento_2025" --detailed

# Comparar com sessões anteriores
python tools/analysis_tools.py --compare-sessions --baseline "evento_2024"

# Gerar relatório para cliente
python tools/visualization_tools.py --client-report --output "relatorio_evento.pdf"
```

### 🎯 **Otimização de Portfolio**

```bash
# Análise para portfólio profissional
python tools/quality_analyzer.py --strategy very_aggressive --portfolio-mode

# Seleção automática das melhores
python tools/analysis_tools.py --top-quality --count 50 --output-dir "portfolio_selected"

# Verificação final
python tools/quality_analyzer.py --validate-selection --dir "portfolio_selected"
```

---

## ℹ️ Informações Técnicas

### 🔬 **Algoritmos Utilizados**

- **Blur Detection**: Variance of Laplacian + Gradient Magnitude
- **Exposure Analysis**: Histogram Statistics + Dynamic Range
- **Person Detection**: MediaPipe + Custom Post-processing
- **Quality Scoring**: Weighted Combination + Machine Learning

### ⚙️ **Requisitos de Performance**

- **CPU**: 2+ cores recomendados
- **RAM**: 4GB mínimo, 8GB recomendado
- **GPU**: Opcional (acelera detecção de pessoas)
- **Armazenamento**: SSD recomendado para melhor I/O

### 📊 **Benchmarks Típicos**

- **Análise Blur**: ~100-200 imagens/minuto
- **Detecção Pessoas**: ~50-100 imagens/minuto
- **Análise Completa**: ~30-60 imagens/minuto
- **Uso de Memória**: ~2-4MB por imagem em processamento

---

*Documentação atualizada para Photo Culling System v2.5*  
*Última revisão: Junho 2025*  
*Para suporte técnico: consulte `tools/README.md`*
