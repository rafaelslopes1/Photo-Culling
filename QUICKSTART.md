# 🚀 Guia de Início Rápido - Photo Culling System

Bem-vindo ao **Photo Culling System v2.5**! Este guia irá te ajudar a começar a usar o sistema em apenas alguns minutos.

---

## ⚡ Configuração Rápida (5 minutos)

### 1️⃣ Pré-requisitos
```bash
# Verificar Python 3.8+
python --version

# Verificar pip
pip --version
```

### 2️⃣ Instalação
```bash
# Clonar o repositório
git clone https://github.com/your-repo/Photo-Culling.git
cd Photo-Culling

# Instalar dependências
pip install -r requirements.txt

# Verificar instalação
python main.py --help
```

### 3️⃣ Primeira Execução
```bash
# Adicionar fotos na pasta data/input/
cp /caminho/suas/fotos/* data/input/

# Processar fotos
python main.py --process-all

# Ver resultados
python main.py --view-results
```

---

## 📸 Fluxo Básico de Uso

### Cenário 1: Processar Fotos Novas
```bash
# 1. Adicionar fotos
cp /suas/fotos/* data/input/

# 2. Análise automática
python main.py --process-batch

# 3. Ver estatísticas
python tools/quality_analyzer.py --summary
```

### Cenário 2: Interface Web para Classificação Manual
```bash
# Iniciar servidor web
python src/web/app.py

# Abrir navegador em http://localhost:5000
# Classificar fotos manualmente
```

### Cenário 3: Análise de Qualidade Detalhada
```bash
# Executar análise completa
python tools/analysis_tools.py --full-analysis

# Gerar relatório visual
python tools/visualization_tools.py --create-report
```

---

## 🎯 Casos de Uso Comuns

### 🔍 Detectar Fotos Borradas
```bash
# Análise de blur automática
python main.py --blur-analysis --strategy balanced

# Ver fotos rejeitadas
python tools/quality_analyzer.py --show-rejected blur
```

### 👥 Análise de Pessoas em Fotos
```bash
# Habilitar detecção de pessoas (config.json)
{
  "processing_settings": {
    "person_analysis": {
      "enabled": true
    }
  }
}

# Processar com análise de pessoas
python main.py --process-all --enable-person-detection
```

### 🎨 Classificação por Qualidade
```bash
# Treinar modelo de qualidade
python tools/ai_prediction_tester.py --train

# Classificar fotos automaticamente
python main.py --ai-classify
```

---

## 📊 Entendendo os Scores

### Blur Score (Pontuação de Nitidez)
- **> 100**: Foto nítida ✅
- **50-100**: Levemente borrada ⚠️
- **< 50**: Muito borrada ❌

### Quality Score (Pontuação de Qualidade)
- **80-100**: Excelente qualidade 🌟
- **60-79**: Boa qualidade ✅
- **40-59**: Qualidade média ⚠️
- **< 40**: Baixa qualidade ❌

### Brightness Score (Brilho)
- **> 200**: Muito clara (superexposta) ⚡
- **100-200**: Bem iluminada ✅
- **50-99**: Escura ⚠️
- **< 50**: Muito escura ❌

*Para detalhes completos, consulte: [ANALYSIS_TOOLS_GUIDE.md](ANALYSIS_TOOLS_GUIDE.md)*

---

## 🛠️ Ferramentas Essenciais

### Análise Rápida
```bash
# Status geral do sistema
python tools/health_check_complete.py

# Análise de qualidade das fotos
python tools/quality_analyzer.py --quick-stats

# Teste do sistema completo
python tools/integration_test.py
```

### Manutenção
```bash
# Limpeza do sistema
python tools/unified_cleanup_tool.py --quick-clean

# Verificar integridade dos dados
python tools/data_quality_cleanup.py --verify

# Manutenção geral
python tools/project_maintenance.py --routine
```

### Visualização
```bash
# Gráficos de qualidade
python tools/visualization_tools.py --quality-charts

# Relatório em HTML
python tools/visualization_tools.py --html-report
```

---

## ⚙️ Configuração Personalizada

### Ajustar Sensibilidade do Blur
```json
// config.json
{
  "processing_settings": {
    "blur_detection_optimized": {
      "strategy": "balanced",  // conservative, balanced, aggressive
      "strategies": {
        "conservative": {"threshold": 50},
        "balanced": {"threshold": 78},
        "aggressive": {"threshold": 145}
      }
    }
  }
}
```

### Habilitar Recursos Avançados
```json
// config.json
{
  "processing_settings": {
    "person_analysis": {
      "enabled": true,
      "min_person_area_ratio": 0.05
    },
    "face_recognition": {
      "enabled": false,
      "threshold": 0.6
    }
  }
}
```

---

## 🚨 Resolução de Problemas

### Erro: "OpenCV não encontrado"
```bash
pip install opencv-python
# ou
pip install opencv-contrib-python
```

### Erro: "Imagem não pode ser carregada"
```bash
# Verificar formatos suportados
python -c "import cv2; print(cv2.getBuildInformation())"

# Converter formato se necessário
python -c "from PIL import Image; img = Image.open('foto.png'); img.save('foto.jpg')"
```

### Performance Lenta
```bash
# Verificar uso de memória
python tools/health_check_complete.py --memory-check

# Processar em lotes menores
python main.py --process-batch --batch-size 50
```

### Banco de Dados Corrompido
```bash
# Verificar integridade
python tools/data_quality_cleanup.py --check-db

# Reconstruir se necessário
python tools/data_quality_cleanup.py --rebuild-db
```

---

## 📈 Próximos Passos

### Para Usuários Básicos
1. ✅ Execute o processamento básico
2. 📊 Explore os relatórios gerados
3. 🖥️ Use a interface web para classificação manual
4. 📖 Leia o guia de análise: [ANALYSIS_TOOLS_GUIDE.md](ANALYSIS_TOOLS_GUIDE.md)

### Para Usuários Avançados
1. 🤖 Configure os modelos de IA
2. ⚙️ Personalize os parâmetros no `config.json`
3. 🔧 Explore as ferramentas de manutenção
4. 📚 Consulte a documentação técnica: [`docs/README.md`](docs/README.md)

### Para Desenvolvedores
1. 🏗️ Estude a arquitetura do projeto
2. 🧪 Execute os testes automatizados
3. 📝 Leia as diretrizes de desenvolvimento
4. 🤝 Consulte o roadmap: [`docs/PROJECT_ROADMAP.md`](docs/PROJECT_ROADMAP.md)

---

## 🆘 Precisa de Ajuda?

- **Documentação Completa**: [`docs/README.md`](docs/README.md)
- **Guia de Ferramentas**: [`tools/README.md`](tools/README.md)
- **Análise e Scores**: [`ANALYSIS_TOOLS_GUIDE.md`](ANALYSIS_TOOLS_GUIDE.md)
- **Histórico de Mudanças**: [`CHANGELOG.md`](CHANGELOG.md)

---

## 🎉 Pronto para Começar!

```bash
# Comando simples para testar tudo
python main.py --demo

# Ou processar suas fotos
python main.py --process-all --verbose
```

**Tempo estimado para primeira execução completa**: 2-5 minutos (dependendo do número de fotos)

---

*Criado em: 27 de dezembro de 2024*  
*Versão do sistema: 2.5.0*
