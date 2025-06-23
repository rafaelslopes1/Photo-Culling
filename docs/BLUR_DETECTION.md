# Detecção de Blur/Desfoque no Photo-Culling

## 📋 Visão Geral

O sistema de detecção de blur implementado no Photo-Culling utiliza o método **Variance of Laplacian** para identificar imagens borradas ou desfocadas automaticamente.

## 🔬 Método Técnico

### Variance of Laplacian
- **Base Científica**: Método desenvolvido por Pech-Pacheco et al. (2000)
- **Princípio**: Mede a variação de intensidade das bordas na imagem
- **Implementação**: `cv2.Laplacian(image, cv2.CV_64F).var()`

### Como Funciona
1. **Conversão**: Imagem convertida para escala de cinza
2. **Convolução**: Aplicação do operador Laplaciano (3x3 kernel)
3. **Cálculo**: Variância da resposta do operador
4. **Classificação**: Comparação com threshold para determinar se é borrada

## 📊 Métricas de Qualidade

### Score de Blur
- **< 20**: Extremamente borrada (considere descartar)
- **20-50**: Muito borrada (qualidade comprometida)
- **50-100**: Levemente borrada (dependendo do threshold)
- **100-200**: Aceitável para uso digital
- **200-500**: Boa nitidez (adequada para impressão)
- **> 500**: Excelente nitidez (ideal para ampliação)

### Análises Complementares
- **Brilho**: Média dos pixels em escala de cinza
- **Contraste**: Desvio padrão dos pixels (spread)
- **Nitidez**: Magnitude do gradiente Sobel
- **Ruído**: Diferença da média local

## 🎯 Configuração de Threshold

### Thresholds Recomendados
- **75**: Detecta apenas casos extremamente borrados
- **100**: Balanceado - recomendado para uso geral
- **150**: Mais rigoroso - ideal para impressão
- **200**: Muito rigoroso - apenas excelente qualidade

### Ajuste por Contexto
- **Fotos de arquivo**: Threshold mais baixo (75-100)
- **Fotos para impressão**: Threshold mais alto (150-200)
- **Fotos para web**: Threshold médio (100-150)

## 🛠️ Uso Prático

### Análise de Imagem Única
```python
from core.image_quality_analyzer import ImageQualityAnalyzer

analyzer = ImageQualityAnalyzer(blur_threshold=100.0)
result = analyzer.analyze_single_image("foto.jpg")

print(f"Blur Score: {result['blur_score']:.2f}")
print(f"Status: {result['blur_status']}")
print(f"Qualidade: {result['quality_rating']}")
```

### Análise de Pasta Completa
```python
analyzer = ImageQualityAnalyzer()
stats = analyzer.analyze_folder("pasta_de_fotos")

print(f"Borradas: {stats['blurry_images']} de {stats['total_images']}")
print(f"Percentual: {stats['blur_percentage']:.1f}%")
```

## 🧰 Ferramentas Incluídas

### 1. `image_quality_analyzer.py`
- Módulo principal de análise
- Implementa detecção de blur e métricas de qualidade
- Armazena resultados em banco SQLite

### 2. `quality_analyzer.py`
- Ferramenta de linha de comando
- Análise em lote com múltiplos thresholds
- Gera relatórios e sugestões de limpeza

### 3. `blur_detection_tester.py`
- Ferramenta de teste e demonstração
- Comparação de thresholds
- Análise de imagens individuais

## 📈 Exemplos de Uso

### Análise Completa da Coleção
```bash
python tools/quality_analyzer.py --analyze
```

### Identificar Imagens Problemáticas
```bash
python tools/quality_analyzer.py --problems
```

### Gerar Sugestões de Limpeza
```bash
python tools/quality_analyzer.py --cleanup
```

### Teste de Imagem Única
```bash
python tools/blur_detection_tester.py --image foto.jpg --threshold 100
```

## 💾 Armazenamento de Dados

### Banco de Dados SQLite
- **Localização**: `data/quality/quality_analysis.db`
- **Campos**: filename, blur_score, is_blurry, timestamps, métricas
- **Queries**: Suporte a busca por critérios de qualidade

### Relatórios JSON
- Análises comparativas com múltiplos thresholds
- Estatísticas detalhadas por sessão
- Histórico de análises

## 🎯 Casos de Uso

### 1. Limpeza de Coleção
- Identificar imagens borradas para remoção
- Economizar espaço de armazenamento
- Manter apenas fotos de qualidade

### 2. Curadoria Automática
- Filtrar apenas imagens nítidas para álbuns
- Priorizar fotos de alta qualidade
- Classificação por níveis de qualidade

### 3. Controle de Qualidade
- Avaliar técnica fotográfica
- Detectar problemas de equipamento
- Monitorar qualidade ao longo do tempo

## 🔧 Personalização

### Ajuste de Threshold
```python
# Para fotos de arquivo/memória
analyzer = ImageQualityAnalyzer(blur_threshold=75.0)

# Para impressão de qualidade
analyzer = ImageQualityAnalyzer(blur_threshold=150.0)
```

### Filtros Customizados
```python
# Buscar apenas casos extremos
blurry_images = analyzer.get_blurry_images()
extreme_cases = [img for img in blurry_images if img['blur_score'] < 20]
```

## 📊 Resultados Esperados

### Coleção Típica de Fotos
- **5-15%** de imagens borradas (threshold 100)
- **2-5%** de casos extremos (score < 20)
- **80-90%** de qualidade aceitável ou superior

### Benefícios
- **Economia de espaço**: 10-20% de redução típica
- **Melhoria de coleção**: Foco em imagens de qualidade
- **Automação**: Reduz tempo de curadoria manual

## 🚀 Integração com Sistema Principal

O sistema de detecção de blur pode ser integrado ao Photo-Culling para:
- **Filtrar** imagens borradas antes da classificação AI
- **Priorizar** imagens nítidas na seleção inteligente
- **Alertar** sobre problemas de qualidade na interface
- **Sugerir** remoção de imagens problemáticas

## 📚 Referências

- Pech-Pacheco et al. (2000): "Diatom autofocusing in brightfield microscopy"
- Pertuz et al. (2013): "Analysis of focus measure operators for shape-from-focus"
- PyImageSearch: "Blur detection with OpenCV"

---

*Sistema implementado com base em técnicas científicas comprovadas e otimizado para uso prático em grandes coleções de imagens.*
