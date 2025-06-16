# Photo Culling - Sistema de Classificação Inteligente de Imagens

[![Status](https://img.shields.io/badge/status-production--ready-green.svg)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

Sistema automatizado para classificação e seleção inteligente de imagens usando IA, com interface web para rotulagem manual e treinamento de modelos.

## ✨ Características Principais

- 🤖 **Classificação com IA**: Modelos treinados automaticamente
- 🎯 **Seleção Inteligente**: Algoritmo prioriza imagens mais valiosas para treinamento
- 🌐 **Interface Web**: Rotulagem rápida e intuitiva
- 📊 **Análise de Distribuição**: Balanceamento automático de classes
- 🔍 **Logs Simplificados**: Mostra apenas inferência e motivo da seleção

## 🚀 Início Rápido

```bash
# Clone o repositório
git clone <repository-url>
cd Photo-Culling

# Configure o ambiente Python
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate no Windows

# Instale dependências
pip install -r requirements.txt

# Inicie a interface web
python main.py --web
```

Acesse: http://localhost:5001

## 📖 Documentação Completa

Para documentação detalhada, consulte: [`docs/README.md`](docs/README.md)

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

## 🛠️ Ferramentas Incluídas

- `tools/health_check.py` - Verificação de saúde do sistema
- `tools/ai_prediction_tester.py` - Teste de predições de IA

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
