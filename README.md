# Photo Culling System

Um sistema completo de organização e rotulagem de imagens com interface web moderna.

## � Funcionalidades Principais

- **Interface Web Intuitiva:** Sistema de rotulagem manual com interface responsiva e moderna
- **Processamento Automático:** Pipeline de análise automática para organização de imagens
- **Detecção de Qualidade:** Avaliação automática baseada em nitidez e brilho
- **Detecção de Duplicatas:** Identificação de imagens duplicadas usando hash perceptual
- **Base de Dados Persistente:** Armazenamento de rótulos e metadados em SQLite

## 📁 Estrutura do Projeto

```
Photo-Culling/
├── web_labeling/           # Interface web de rotulagem manual
│   ├── app.py             # Servidor Flask
│   ├── templates/         # Templates HTML
│   │   └── index.html     # Interface principal
│   ├── data/              # Base de dados e backups
│   │   ├── labels.db      # SQLite database
│   │   └── labels.json    # Backup JSON
│   └── requirements.txt   # Dependências do web app
├── image_culling.py       # Pipeline de processamento automático
├── advanced_*.py          # Módulos de detecção avançada
├── config.json           # Configurações do sistema
├── input/                # Pasta de imagens de entrada
└── requirements.txt      # Dependências principais
```

## � Como Usar

### Interface Web (Recomendado)

1. **Instalar dependências:**
```bash
cd web_labeling
pip install -r requirements.txt
```

2. **Executar o servidor:**
```bash
python app.py
```

3. **Acessar a interface:**
   - Abra seu navegador em `http://localhost:5002`
   - Use os atalhos de teclado para rotulagem rápida
   - Pressione `I` para mostrar/ocultar informações da imagem

### Pipeline Automático

1. **Instalar dependências:**
```bash
pip install -r requirements.txt
```

2. **Executar processamento:**
```bash
python image_culling.py input/ output/
```

## ⌨️ Atalhos de Teclado (Interface Web)

- **Navegação:**
  - `←` ou `A`: Imagem anterior
  - `→` ou `D`: Próxima imagem
  - `Space`: Próxima imagem

- **Rotulagem:**
  - `1-6`: Aplicar rótulos 1-6
  - `Q`: Rejeitar (qualidade ruim)
  - `R`: Rejeitar (conteúdo inapropriado)

- **Interface:**
  - `I`: Mostrar/ocultar informações da imagem
  - `Esc`: Fechar modais

## 🛠️ Configuração

Personalize o comportamento através do arquivo `config.json`:

```json
{
  "processing_settings": {
    "blur_threshold": 100,
    "brightness_threshold": 50,
    "nsfw_threshold": 0.7,
    "quality_score_weights": {
      "sharpness": 1.0,
      "brightness": 1.0
    },
    "image_extensions": [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"]
  },
  "output_folders": {
    "selected": "selected",
    "duplicates": "duplicates", 
    "blurry": "blurry",
    "low_light": "low_light",
    "nsfw": "nsfw",
    "failed": "failed"
  }
}
```

## 📋 Requisitos

- Python 3.7+
- OpenCV (`opencv-python`)
- Pillow
- ImageHash
- NumPy
- Flask (para interface web)
- SQLite3

## 💡 Dicas

- Use a interface web para rotulagem manual precisa
- Configure os thresholds no `config.json` conforme suas necessidades
- Mantenha backups regulares da base de dados de rótulos
- Use atalhos de teclado para acelerar o processo de rotulagem

## 📈 Estrutura de Saída (Pipeline Automático)

```
output/
├── selected/     📸 Imagens de alta qualidade (ranqueadas)
├── duplicates/   🔄 Imagens duplicadas detectadas
├── blurry/       💫 Imagens desfocadas
├── low_light/    🌑 Imagens muito escuras
├── nsfw/         🔞 Conteúdo inapropriado (se habilitado)
└── failed/       ❌ Imagens que falharam no processamento
```

## License

This project is licensed under the MIT License.
