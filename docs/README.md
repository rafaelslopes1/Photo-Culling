# Photo-Culling System - Documentação

## Visão Geral
Sistema automatizado para classificação de qualidade de imagens com interface web para rotulagem manual e treinamento de modelos de IA.

## Arquitetura do Sistema

### Estrutura do Projeto
```
Photo-Culling/
├── src/
│   ├── core/           # Módulos principais
│   │   ├── ai_classifier.py     # Classificador de IA
│   │   ├── feature_extractor.py # Extração de características
│   │   └── image_processor.py   # Processamento de imagens
│   ├── web/            # Interface web
│   │   ├── app.py              # Aplicação Flask
│   │   └── templates/          # Templates HTML
│   └── utils/          # Utilitários
│       ├── config_manager.py   # Gerenciamento de configuração
│       └── data_utils.py       # Utilitários de dados
├── data/
│   ├── input/          # Imagens de entrada
│   ├── labels/         # Banco de dados de rótulos
│   ├── features/       # Banco de dados de características
│   └── models/         # Modelos treinados
├── docs/               # Documentação
├── tools/              # Ferramentas auxiliares
└── main.py            # Ponto de entrada principal
```

## Funcionalidades

### Seleção Inteligente de Imagens
- **Priorização de classes sub-representadas**: Identifica classes com poucos exemplos
- **Análise de incerteza**: Seleciona casos onde o modelo está menos confiante
- **Balanceamento automático**: Contribui para um dataset mais equilibrado

### Logs Simplificados
```
🎯 SELEÇÃO: IMG_0234.JPG
🤖 Algoritmo inferiu: 31.2% chance de ser 'quality_2'
📊 Motivo da sugestão: Classe tem apenas 5 exemplos (sub-representada)
──────────────────────────────────────────────────
```

### Classes de Classificação

#### Qualidade (1-5 estrelas)
- `quality_1`: ⭐ Qualidade Muito Baixa
- `quality_2`: ⭐⭐ Qualidade Baixa
- `quality_3`: ⭐⭐⭐ Qualidade Média
- `quality_4`: ⭐⭐⭐⭐ Qualidade Boa
- `quality_5`: ⭐⭐⭐⭐⭐ Qualidade Excelente

#### Rejeição
- `reject_dark`: 🌑 Muito Escura
- `reject_light`: ☀️ Muito Clara
- `reject_blur`: 😵‍💫 Muito Borrada
- `reject_cropped`: ✂️ Cortada/Incompleta
- `reject_other`: ❌ Outros Problemas

## Uso do Sistema

### Iniciando a Interface Web
```bash
python main.py --web-interface --selection-mode smart
```

### Extraindo Características
```bash
python main.py --extract-features
```

### Treinando Modelo
```bash
python main.py --train-model
```

### Classificação Automática
```bash
python main.py --classify
```

## Estado do Projeto

✅ **Concluído**:
- Estrutura modular limpa
- Interface web funcional
- Seleção inteligente implementada
- Logs simplificados
- Sistema de rotulagem completo
- Integração IA funcional

## Última Atualização: 15 de Junho de 2025
- Implementação de logs simplificados
- Limpeza de código duplicado
- Consolidação da documentação
- Otimização da seleção inteligente
