# 📸 Photo Culling System v2.0

[![Status](https://img.shields.io/badge/status-production--ready-green.svg)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()
[![Blur Detection](https://img.shields.io/badge/Blur%20Detection-Optimized-brightgreen.svg)]()

**Sistema inteligente de classificação e curadoria de fotografias com detecção otimizada de blur**

## 🆕 Novidades v2.0 - Sistema de Blur Detection Otimizado

### ✨ Principais Funcionalidades
- 🎯 **Detecção Inteligente de Blur** com validação supervisionada
- 🎚️ **Estratégias Configuráveis** para diferentes casos de uso
- 📊 **Análise de Qualidade Avançada** baseada em 440 exemplos rotulados
- 🔄 **Pipeline Automatizado** de classificação e organização
- 🌐 **Interface Web** para rotulação manual e treinamento
- 🤖 **Aprendizado Supervisionado** para otimização contínua

## 🎯 Estratégias de Blur Detection

| Estratégia | Threshold | Caso de Uso | Taxa Remoção |
|------------|-----------|-------------|--------------|
| `conservative` | 50 | Arquivos pessoais | ~30% |
| `balanced` | 78 | **Uso geral (padrão)** | ~50% |
| `aggressive` | 145 | Portfólio profissional | ~70% |
| `very_aggressive` | 98 | Exposições/impressão | ~60% |

## 🚀 Como Usar

### Classificação Automática com Blur Detection Otimizado
```bash
# Processar imagens com sistema otimizado
python main.py --classify --input-dir data/input

# Análise de qualidade detalhada
python tools/quality_analyzer.py --analyze

# Manutenção automática do projeto
python tools/project_maintenance.py
```

### Interface Web e Treinamento
```bash
# Interface web para rotulação
python main.py --web-interface --port 5001

# Extrair características
python main.py --extract-features --input-dir data/input

# Treinar modelo de IA
python main.py --train-model
```

### Configuração do Sistema de Blur Detection

Edite `config.json` para ajustar a estratégia:
```json
{
  "processing_settings": {
    "blur_detection_optimized": {
      "enabled": true,
      "strategy": "balanced",  // conservative, balanced, aggressive, very_aggressive
      "debug": false
    }
  }
}
```

# Inicie a interface web
python main.py --web
```

Acesse: http://localhost:5001

## 📖 Documentação Completa

### 📚 Guias Principais
- [`docs/README.md`](docs/README.md) - Documentação técnica completa
- [`ANALYSIS_TOOLS_GUIDE.md`](ANALYSIS_TOOLS_GUIDE.md) - **🆕 Guia de ferramentas de análise e scores**
- [`tools/README.md`](tools/README.md) - Ferramentas de manutenção e utilitários

### 📋 Guias Específicos
- [`QUICKSTART.md`](QUICKSTART.md) - Guia de início rápido
- [`CHANGELOG.md`](CHANGELOG.md) - Histórico de mudanças  
- [`docs/SMART_SELECTION.md`](docs/SMART_SELECTION.md) - Como funciona a seleção inteligente

## 🏗️ Estrutura do Projeto

```
Photo-Culling/
├── src/                    # Código fonte modular
│   ├── core/              # Processamento de IA e features
│   ├── web/               # Interface web
│   └── utils/             # Utilitários e configuração
├── data/                  # Dados do projeto
│   ├── input/             # Imagens para classificar
│   ├── labels/            # Rótulos e anotações
│   ├── features/          # Features extraídas
│   └── models/            # Modelos treinados
├── docs/                  # Documentação
└── tools/                 # Ferramentas auxiliares
```

## 🤖 Sistema de Seleção Inteligente

O algoritmo de seleção inteligente:

1. **Analisa a distribuição** de classes existentes
2. **Identifica classes sub-representadas** 
3. **Usa IA para prever** quais imagens podem pertencer a essas classes
4. **Prioriza imagens** com maior valor de treinamento
5. **Registra apenas** a inferência e o motivo da escolha

Exemplo de log simplificado:
```
🎯 SELEÇÃO: IMG_0123.JPG
🤖 Algoritmo inferiu: 65% chance de ser 'portrait'
📊 Motivo da sugestão: Classe tem apenas 8 exemplos (sub-representada)
```

## 📈 Estatísticas

- **Tempo médio de classificação**: ~2-3 segundos por imagem
- **Precisão típica**: 85-95% após 50+ exemplos por classe
- **Redução de tempo**: 70% comparado à seleção manual

## 🛠️ Ferramentas de Manutenção e Análise

### 🔧 Manutenção Automatizada
- `tools/project_maintenance.py` - Monitoramento e manutenção automática do projeto
- `tools/unified_cleanup_tool.py` - Ferramenta unificada de análise e limpeza
- `tools/data_quality_cleanup.py` - Limpeza especializada de dados de qualidade

### 📊 Análise e Qualidade
- `tools/quality_analyzer.py` - Análise detalhada de qualidade de imagens com scores
- `tools/analysis_tools.py` - Ferramentas estatísticas e métricas avançadas
- `tools/visualization_tools.py` - Visualizações e gráficos de análise

### 🤖 Testes de IA
- `tools/ai_prediction_tester.py` - Validação de predições e modelos de IA
- `tools/face_recognition_test.py` - Testes específicos de reconhecimento facial

### 🚀 Uso das Ferramentas
```bash
# Manutenção diária
python tools/project_maintenance.py

# Análise de qualidade com scores detalhados
python tools/quality_analyzer.py --analyze

# Limpeza completa do projeto
python tools/unified_cleanup_tool.py

# Análise estatística avançada
python tools/analysis_tools.py
```

## 📝 Guias Rápidos

- [`QUICKSTART.md`](QUICKSTART.md) - Guia de início rápido
- [`CHANGELOG.md`](CHANGELOG.md) - Histórico de mudanças
- [`docs/SMART_SELECTION.md`](docs/SMART_SELECTION.md) - Como funciona a seleção inteligente
- [`docs/LOGS_SIMPLIFICADOS.md`](docs/LOGS_SIMPLIFICADOS.md) - Sistema de logs

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -am 'feat: adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo LICENSE para detalhes.

---

**Status**: ✅ Pronto para produção | **Última atualização**: Junho 2025

## 🆕 NOVO: Sistema Otimizado de Blur Detection

### ✅ Integração Completa (Junho 2025)
O sistema agora inclui **detecção avançada de desfoque** com validação supervisionada:

- 🎯 **4 estratégias otimizadas** para diferentes cenários
- 📊 **Validação com 440 imagens** rotuladas manualmente  
- ⚙️ **Configuração flexível** via `config.json`
- 🔍 **Análise detalhada** com categorização em 5 níveis

#### 🎚️ Estratégias Disponíveis
| Estratégia | Threshold | Uso Recomendado |
|------------|-----------|-----------------|
| `conservative` | 50 | Arquivo pessoal/histórico |
| `balanced` | 78 | **Uso geral (padrão)** |
| `aggressive` | 145 | Portfólio profissional |
| `very_aggressive` | 98 | Exposições/impressão |

#### 🚀 Como Usar
```bash
# Classificação com blur detection otimizado
python main.py --classify --input-dir data/input

# Análise de qualidade com ferramentas atualizadas
python tools/quality_analyzer.py --analyze

# Manutenção e limpeza do projeto
python tools/project_maintenance.py --clean
```

---
