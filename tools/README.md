# Tools Directory - Photo Culling System v2.5

Este diretório contém utilitários e ferramentas para desenvolvimento, teste e análise do sistema.

## 🔧 Utilitários Principais

### **unified_test_suite.py** ⭐
**Suite de testes unificada e principal**
- Testa todos os componentes do sistema (GPU otimizado)
- Verifica saúde do sistema, detecção de pessoas, extração de features
- Análise de performance com otimização Mac M3
- **USO**: `python tools/unified_test_suite.py`

### **system_demo.py**
**Demonstração interativa do sistema**
- Mostra funcionamento completo do pipeline
- Testa configurações de blur detection
- Processamento de imagens de exemplo
- **USO**: `python tools/system_demo.py`

## 📊 Ferramentas de Análise

### **analysis_tools.py**
**Ferramentas de análise estatística**
- Análise de performance de algoritmos
- Estatísticas de qualidade de imagem
- Comparação de resultados

### **quality_analyzer.py**
**Analisador de qualidade de imagem**
- Métricas detalhadas de qualidade
- Análise de blur, exposição, composição
- Relatórios de qualidade

### **visualization_tools.py**
**Ferramentas de visualização**
- Gráficos de análise de dados
- Visualização de resultados
- Plots de performance

## 🤖 Ferramentas de AI

### **ai_prediction_tester.py**
**Testador de predições de AI**
- Testa acurácia dos modelos de ML
- Validação de classificadores
- Métricas de performance de AI

### **generate_final_report.py**
**Gerador de relatórios finais**
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
