# Sistema Web de Rotulagem de Imagens

## 🎯 Visão Geral

Sistema web moderno e eficiente para rotulagem rápida de imagens usando o teclado. Ideal para classificação de qualidade de fotos e preparação de datasets para treino de IA.

## ✨ Características

### ⌨️ Controles por Teclado

**Qualidade (1-5 estrelas):**
- `1` - ⭐ Qualidade Muito Baixa
- `2` - ⭐⭐ Qualidade Baixa  
- `3` - ⭐⭐⭐ Qualidade Média
- `4` - ⭐⭐⭐⭐ Qualidade Boa
- `5` - ⭐⭐⭐⭐⭐ Qualidade Excelente

**Rejeição por Problemas:**
- `D` - 🌑 Muito Escura
- `L` - ☀️ Muito Clara
- `B` - 😵‍💫 Muito Borrada
- `C` - ✂️ Cortada/Incompleta
- `X` - ❌ Outros Problemas

**Navegação:**
- `←` - Imagem anterior
- `→` - Próxima imagem
- `↑` - Primeira imagem
- `↓` - Última imagem

### 🚀 Funcionalidades

1. **Rotulagem Rápida**: Pressione uma tecla e automaticamente avança para próxima imagem
2. **Feedback Visual**: Mostra o rótulo aplicado e progresso em tempo real
3. **Sobrescrita**: Pode alterar rótulos de imagens já classificadas
4. **Banco de Dados**: SQLite para persistência e backup JSON automático
5. **Estatísticas**: Acompanha progresso e distribuição dos rótulos
6. **Interface Moderna**: Design responsivo e intuitivo

## 🛠️ Instalação e Uso

### 1. Instalar Dependências
```bash
cd web_labeling
pip install -r requirements.txt
```

### 2. Configurar Pasta de Imagens
Por padrão, o sistema lê imagens da pasta `../input`. Para usar outra pasta, edite a variável `IMAGES_DIR` no arquivo `app.py`.

### 3. Executar o Servidor
```bash
python app.py
```

### 4. Acessar a Interface
Abra seu navegador em: `http://localhost:5000`

## 📊 Dados Gerados

### Banco de Dados SQLite (`data/labels.db`)
- Tabela `labels`: Todos os rótulos com timestamps
- Tabela `sessions`: Controle de sessões de rotulagem

### Backup JSON (`data/labels.json`)
Arquivo de backup legível com todos os rótulos aplicados.

### Estrutura dos Dados
```json
{
  "IMG_001.jpg": {
    "type": "quality",
    "score": 4,
    "rejection_reason": null,
    "timestamp": "2025-06-09T14:30:00"
  },
  "IMG_002.jpg": {
    "type": "rejection", 
    "score": null,
    "rejection_reason": "blur",
    "timestamp": "2025-06-09T14:31:00"
  }
}
```

## 🎯 Fluxo de Trabalho

1. **Inicie o servidor** e acesse a interface web
2. **Navegue pelas imagens** usando as setas do teclado
3. **Classifique rapidamente** pressionando 1-5 para qualidade ou D/L/B/C/X para rejeição
4. **Acompanhe o progresso** na barra superior
5. **Continue até terminar** - o sistema salva automaticamente

## 🔄 Integração com IA

Os dados gerados estão prontos para:
- **Treino de modelos** de classificação de qualidade
- **Análise estatística** de padrões nas imagens
- **Filtragem automática** baseada nos critérios definidos
- **Export para frameworks** como TensorFlow/PyTorch

## 🎨 Interface

- **Design moderno** com gradientes e efeitos visuais
- **Feedback imediato** ao aplicar rótulos
- **Estatísticas em tempo real** de progresso
- **Responsiva** para diferentes tamanhos de tela
- **Atalhos visuais** mostrando todas as teclas disponíveis

## 🔧 Personalização

Para modificar os rótulos disponíveis, edite as variáveis no `app.py`:
- `QUALITY_KEYS`: Mapeamento de teclas para qualidade
- `REJECTION_KEYS`: Mapeamento de teclas para rejeição

## 📈 Vantagens sobre GUI Desktop

✅ **Mais Rápido**: Interface web é mais responsiva
✅ **Multiplataforma**: Funciona em qualquer sistema operacional  
✅ **Sem Problemas de Display**: Não depende do sistema de janelas
✅ **Mais Moderno**: Interface bonita e profissional
✅ **Escalável**: Pode ser acessado remotamente
✅ **Backup Automático**: SQLite + JSON para segurança dos dados
