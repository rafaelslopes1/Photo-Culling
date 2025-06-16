# 📋 Lógica de Seleção da Próxima Foto - Photo Culling System 2.0

## 🎯 Resumo da Lógica Atual

O sistema implementa uma **lógica sequencial e determinística** para seleção da próxima foto a ser rotulada, baseada em **ordem alfabética** e **status de rotulagem**.

---

## 🔍 Análise Detalhada da Lógica

### 1. **Carregamento Inicial das Imagens**

```python
def load_image_list(self):
    """Carrega lista de imagens em ordem determinística"""
    # Busca por extensões suportadas
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    
    # Ordem determinística (alfabética) para evitar repetições
    image_names = sorted([img.name for img in images])
```

**📌 Características:**
- ✅ **Ordem Determinística**: Sempre a mesma ordem entre sessões
- ✅ **Alfabética**: Baseada no nome do arquivo (IMG_0001.JPG, IMG_0002.JPG, etc.)
- ✅ **Extensões Múltiplas**: Suporta vários formatos de imagem
- ✅ **Case-Insensitive**: Considera .jpg e .JPG

### 2. **Identificação de Imagens Não Rotuladas**

```python
def get_unlabeled_images(self):
    """Retorna lista de imagens não rotuladas"""
    # Consulta banco de dados para imagens já rotuladas
    cursor.execute('SELECT filename FROM labels')
    labeled_files = {row[0] for row in cursor.fetchall()}
    
    # Filtra apenas as não rotuladas, mantendo a ordem original
    unlabeled = [img for img in self.image_list if img not in labeled_files]
```

**📌 Características:**
- ✅ **Consulta em Tempo Real**: Verifica banco de dados a cada requisição
- ✅ **Preserva Ordem**: Mantém ordem alfabética das imagens
- ✅ **Eficiente**: Usa set para lookup rápido de imagens rotuladas

### 3. **Seleção da Próxima Imagem**

#### 🎮 **Método 1: API `/api/next_image`**
```python
def next_image():
    """Obtém próxima imagem para rotular"""
    unlabeled = self.get_unlabeled_images()
    
    if not unlabeled:
        return {'finished': True}
    
    filename = unlabeled[0]  # SEMPRE A PRIMEIRA NÃO ROTULADA
```

#### 🎮 **Método 2: API `/api/first-unlabeled`**
```python
def first_unlabeled():
    """Primeira imagem não rotulada"""
    unlabeled = self.get_unlabeled_images()
    return unlabeled[0]  # SEMPRE A PRIMEIRA NÃO ROTULADA
```

**📌 Características:**
- ✅ **Sempre a Primeira**: Seleciona sempre a primeira imagem não rotulada da lista
- ✅ **Sequencial**: Segue ordem alfabética dos nomes de arquivo
- ✅ **Consistente**: Mesmo comportamento em ambas as APIs

---

## 🔄 Fluxo de Trabalho da Seleção

### 1. **Inicialização do Sistema**
```
1. Carrega todas as imagens do diretório data/input/
2. Ordena alfabeticamente (IMG_0001.JPG, IMG_0002.JPG, ...)
3. Armazena em self.image_list
```

### 2. **Quando Usuário Pede "Próxima Imagem"**
```
1. Consulta banco de dados: quais já foram rotuladas?
2. Filtra lista original: remove as rotuladas
3. Retorna unlabeled[0]: primeira não rotulada
```

### 3. **Exemplo Prático**

Imagine que temos estas imagens:
```
Lista Original (alfabética):
- IMG_0001.JPG  ← rotulada
- IMG_0002.JPG  ← rotulada  
- IMG_0003.JPG  ← não rotulada
- IMG_0004.JPG  ← não rotulada
- IMG_0005.JPG  ← rotulada
```

**Resultado da seleção**: `IMG_0003.JPG` (primeira não rotulada)

---

## 🎯 Comportamentos Importantes

### ✅ **Vantagens da Lógica Atual**
1. **Determinística**: Sempre previsível
2. **Sequencial**: Progresso linear através das imagens
3. **Eficiente**: Consulta rápida ao banco de dados
4. **Consistente**: Mesmo comportamento entre sessões
5. **Simples**: Fácil de entender e depurar

### ⚠️ **Limitações da Lógica Atual**
1. **Não Random**: Não há randomização
2. **Não Prioritizada**: Não considera qualidade ou dificuldade
3. **Não Inteligente**: Não usa IA para priorizar
4. **Sequencial Rígida**: Sempre ordem alfabética

---

## 🔧 Possíveis Melhorias Futuras

### 🎲 **Opção 1: Seleção Inteligente com IA**
```python
def get_next_intelligent_image():
    """Seleciona próxima imagem baseada em critérios inteligentes"""
    unlabeled = get_unlabeled_images()
    
    # Priorizar por:
    # 1. Imagens que IA tem baixa confiança
    # 2. Imagens com características interessantes
    # 3. Balanceamento de classes
```

### 🎲 **Opção 2: Seleção Randomizada**
```python
def get_random_unlabeled():
    """Seleciona imagem aleatória não rotulada"""
    unlabeled = get_unlabeled_images()
    return random.choice(unlabeled)
```

### 🎲 **Opção 3: Seleção por Estratégia**
```python
def get_strategic_image(strategy='sequential'):
    """Seleciona baseado em estratégia configurável"""
    strategies = {
        'sequential': get_sequential_image,
        'random': get_random_image,
        'ai_guided': get_ai_guided_image,
        'balanced': get_balanced_image
    }
    return strategies[strategy]()
```

---

## 📊 Status Atual: SEQUENCIAL ALFABÉTICA

**🎯 Lógica Atual**: Sempre a **primeira imagem não rotulada** em **ordem alfabética**

**✅ Funciona perfeitamente** para:
- Progressão linear através do dataset
- Garantia de que todas as imagens serão eventualmente rotuladas
- Reprodutibilidade entre sessões
- Simplicidade de uso e debugging

**🔌 Para usar**: A interface web já implementa esta lógica automaticamente!
