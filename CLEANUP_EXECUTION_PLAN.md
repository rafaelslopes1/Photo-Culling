# 🧹 PLANO DE LIMPEZA DO PROJETO - Photo Culling

## 📋 ANÁLISE DE ARQUIVOS

### 🗑️ ARQUIVOS PARA REMOÇÃO

#### Root - Documentos Duplicados/Obsoletos:
- `AI_SYSTEM_STATUS.md`
- `BUG_FIX_REPORT.md`
- `BUG_REPORT.md`
- `CHANGELOG.md`
- `CLEANUP_PLAN.md`
- `COMO_FUNCIONA_IA.md`
- `CORRECAO_SPACE.md`
- `IMAGE_BACKUP_GUIDE.md`
- `PLANO_DE_ACAO.md`
- `QUICKSTART.md`
- `REORGANIZATION_PLAN.md`
- `SISTEMA_COMPLETO.md`
- `SMART_SELECTION_LOGS.md`
- `STATUS_REPORT.md`
- `TROUBLESHOOT_IA.md`

#### Root - Scripts Duplicados/Obsoletos:
- `ai_classifier.py` (duplicado - existe em src/core/)
- `analyze_blur_rejections.py` (primeira versão - obsoleta)
- `analyze_blur_rejections_specific.py` (versão intermediária)
- `analyze_hybrid_strategy.py` (análise concluída)
- `auto_optimization_system.py` (obsoleto)
- `clean_labels.py` (utilitário temporário)
- `config_backup.json` (backup)
- `config_clean.json` (temporário)
- `detailed_model_analyzer.py` (obsoleto)
- `enhanced_feature_extractor.py` (obsoleto)
- `enhanced_features.py` (obsoleto)
- `extract_labeled_features.py` (obsoleto)
- `feature_extractor.py` (duplicado - existe em src/core/)
- `generate_final_report.py` (obsoleto)
- `image_backup.py` (utilitário temporário)
- `inspect_database.py` (utilitário temporário)
- `model_quality_analyzer.py` (obsoleto)
- `requirements_new.txt` (duplicado)
- `smart_labeling_system.py` (obsoleto)
- `smart_threshold_optimization.py` (análise concluída)
- `test-labels.py` (teste temporário)
- `test_app_bug.py` (teste temporário)
- `test_custom_threshold.py` (teste específico - mover para tools/)
- `test_fixed_app.py` (teste temporário)
- `test_labeled_appearing.py` (teste temporário)
- `test_parallel_features.py` (teste temporário)
- `test_practical_strategies.py` (teste específico - mover para tools/)
- `test_randomization.py` (teste temporário)
- `verify_integration.py` (obsoleto)

#### Docs - Documentos Duplicados:
- `FINAL_FIXES.md`
- `IMAGE_LOADING_FIX.md`
- `IMPLEMENTACAO_CONCLUIDA.md`
- `INTEGRATION_COMPLETE.md` (duplicado do INTEGRATION_STATUS_FINAL.md)
- `LOGS_DETALHADOS.md`
- `LOGS_MELHORADOS.md`
- `LOGS_SIMPLIFICADOS.md`
- `PROBLEMA_RESOLVIDO_ROTULAGEM.md`
- `PROJECT_COMPLETE.md`
- `REORGANIZATION_COMPLETE.md`
- `SELECTION_LOGIC.md`
- `SMART_SELECTION.md`
- `TOTAL_SUCCESS.md`

#### Web_labeling - Diretório Obsoleto:
- Todo o diretório `web_labeling/` (funcionalidade movida para src/web/)

### 🔄 ARQUIVOS PARA MOVER/RENOMEAR

#### Para tools/:
- `demo_integrated_system.py` → `tools/demo_system.py`
- `final_integration_test.py` → `tools/integration_test.py`
- `system_health_check.py` → `tools/health_check_complete.py`
- `test_custom_threshold.py` → `tools/threshold_tester.py`
- `test_practical_strategies.py` → `tools/strategy_tester.py`

### 📁 DIRETÓRIOS PARA LIMPEZA

#### data/quality/:
- Consolidar configurações de blur em um arquivo único

## 🎯 ESTRUTURA FINAL DESEJADA

```
Photo-Culling/
├── src/
│   ├── core/           # Código principal
│   ├── utils/          # Utilitários
│   └── web/           # Interface web
├── data/
│   ├── input/         # Imagens de entrada
│   ├── labels/        # Base de dados de rótulos
│   ├── models/        # Modelos treinados
│   ├── features/      # Características extraídas
│   └── quality/       # Configurações de qualidade
├── tools/             # Scripts de análise e teste
├── docs/              # Documentação essencial (3-4 arquivos)
├── config.json        # Configuração principal
├── main.py           # Entrada principal
├── requirements.txt   # Dependências
└── README.md         # Documentação principal
```

## ✅ ARQUIVOS ESSENCIAIS A MANTER

### Root:
- `main.py` (entrada principal)
- `config.json` (configuração principal) 
- `requirements.txt` (dependências)
- `README.md` (documentação principal)

### Docs:
- `BLUR_DETECTION.md` (documentação técnica)
- `BLUR_ANALYSIS_EXECUTIVE_SUMMARY.md` (resumo executivo)
- `INTEGRATION_STATUS_FINAL.md` (status final)

### Tools:
- Scripts de análise essenciais (consolidados)
