# Tools Directory - Photo Culling System v2.5

Este diretório contém utilitários consolidados e ferramentas para manutenção, análise e limpeza do sistema.

## 🔧 Utilitários Principais

### 🛠️ **Manutenção e Limpeza**

#### **project_maintenance.py** ⭐
**Sistema automatizado de manutenção do projeto**
- Monitoramento de arquivos grandes e cache
- Limpeza automática de diretórios temporários
- Relatórios detalhados com recomendações
- **USO**: `python tools/project_maintenance.py [--clean]`

#### **unified_cleanup_tool.py** 🧹
**Ferramenta unificada de análise e limpeza**
- Análise completa da estrutura do projeto
- Detecção de duplicatas e arquivos redundantes
- Limpeza segura com modo simulação
- **USO**: `python tools/unified_cleanup_tool.py [--analyze] [--dry-run]`

#### **data_quality_cleanup.py** 📊
**Limpeza específica de dados de qualidade**
- Remove análises antigas e temporárias
- Consolida relatórios essenciais
- Otimização de espaço em disco
- **USO**: `python tools/data_quality_cleanup.py [--days N]`

### 📊 **Análise e Qualidade**

#### **quality_analyzer.py**
**Analisador de qualidade de imagem**
- Métricas detalhadas de qualidade
- Análise de blur, exposição, composição
- Sugestões de limpeza baseadas na qualidade
- **USO**: `python tools/quality_analyzer.py --analyze`

#### **analysis_tools.py**
**Ferramentas de análise estatística**
- Análise de performance de algoritmos
- Estatísticas de qualidade de imagem
- Comparação de resultados

#### **visualization_tools.py**
**Ferramentas de visualização**
- Gráficos de análise de dados
- Visualização de resultados
- Plots de performance

### 🤖 **Testes e IA**

#### **ai_prediction_tester.py**
**Testador de predições de AI**
- Testa acurácia dos modelos de ML
- Validação de classificadores
- Métricas de performance de AI

#### **face_recognition_test.py**
**Testes de reconhecimento facial**
- Validação do sistema de reconhecimento
- Testes de acurácia facial
- Debug de problemas de detecção

## 🚀 Fluxos de Trabalho Recomendados

### 📅 **Manutenção Diária**
```bash
# Verificação de saúde geral
python tools/project_maintenance.py

# Análise sem modificar arquivos
python tools/unified_cleanup_tool.py --analyze
```

### 🗓️ **Manutenção Semanal**
```bash
# Limpeza completa
python tools/project_maintenance.py --clean
python tools/unified_cleanup_tool.py

# Análise de qualidade
python tools/quality_analyzer.py --analyze
```

### 📆 **Manutenção Mensal**
```bash
# Limpeza de dados antigos
python tools/data_quality_cleanup.py --days 30

# Testes completos de IA
python tools/ai_prediction_tester.py
```

## 📈 Scripts Consolidados

### ✅ **Ferramentas Ativas**
- `project_maintenance.py` - Manutenção automatizada
- `unified_cleanup_tool.py` - Limpeza unificada (novo)
- `data_quality_cleanup.py` - Limpeza de dados (novo)
- `quality_analyzer.py` - Análise de qualidade
- `analysis_tools.py` - Ferramentas de análise
- `visualization_tools.py` - Visualizações
- `ai_prediction_tester.py` - Testes de IA
- `face_recognition_test.py` - Testes de reconhecimento

### ❌ **Scripts Removidos (Consolidados)**
- `project_cleanup_analysis.py` → integrado no `unified_cleanup_tool.py`
- `execute_cleanup.py` → integrado no `unified_cleanup_tool.py`
- `advanced_cleanup.py` → integrado no `unified_cleanup_tool.py`
- Scripts de teste redundantes → consolidados
- Compilação de resultados de análise
- Relatórios de performance
- Documentação automática

## 🚀 Como Usar

### Teste Rápido do Sistema
```bash
python tools/unified_test_suite.py
```

### Demonstração Completa
```bash
python tools/system_demo.py
```

### Análise de Qualidade
```bash
python tools/quality_analyzer.py
```

## 📋 Ordem de Execução Recomendada

1. **unified_test_suite.py** - Verificar se tudo está funcionando
2. **system_demo.py** - Ver o sistema em ação
3. **quality_analyzer.py** - Análise detalhada de qualidade
4. **analysis_tools.py** - Análises estatísticas avançadas
5. **ai_prediction_tester.py** - Validar modelos de AI

## ⚡ Otimizações

Todas as ferramentas estão otimizadas para:
- **Mac M3 GPU**: Aceleração automática via Metal
- **Logging Silencioso**: Supressão de mensagens técnicas
- **Performance**: Processamento rápido e eficiente

## 🔧 Manutenção

Para manter as ferramentas atualizadas:
- Executar `unified_test_suite.py` diariamente
- Verificar relatórios de `quality_analyzer.py` semanalmente
- Atualizar configurações conforme necessário

## 🎯 Estrutura Final do Diretório

```
tools/
├── README.md                    # 📋 Esta documentação
├── project_maintenance.py       # 🔧 Manutenção automatizada
├── unified_cleanup_tool.py      # 🧹 Limpeza unificada
├── data_quality_cleanup.py      # 📊 Limpeza de dados
├── quality_analyzer.py          # 🔍 Análise de qualidade
├── analysis_tools.py            # 📈 Ferramentas de análise
├── visualization_tools.py       # 📊 Visualizações
├── ai_prediction_tester.py      # 🤖 Testes de IA
└── face_recognition_test.py     # 👤 Testes de reconhecimento
```

## 💡 Dicas de Uso

### Para Desenvolvedores
- Use `--analyze` para verificar antes de fazer mudanças
- Use `--dry-run` para simular limpezas
- Execute manutenção diária para manter projeto limpo

### Para Análise de Dados
- `quality_analyzer.py` para insights sobre imagens
- `visualization_tools.py` para gráficos
- `analysis_tools.py` para estatísticas detalhadas

### Para Teste de IA
- `ai_prediction_tester.py` para validar modelos
- `face_recognition_test.py` para debug facial

---

*Documentação atualizada após consolidação de ferramentas*  
*Última atualização: Junho 2025*  
*Versão: 2.5 - Ferramentas Consolidadas*
