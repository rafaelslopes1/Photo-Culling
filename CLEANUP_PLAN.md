# Plano de Limpeza do Projeto Photo Culling

## Scripts e arquivos duplicados/temporários para REMOVER:

### 1. Scripts de análise e teste temporários (raiz)
- `analyze_blur_rejections.py` → Mover para tools/
- `analyze_multi_person_detection.py` → Consolidar no tools/
- `debug_serialization.py` → Remover (temporário)
- `showcase_multi_person_detection.py` → Consolidar no tools/
- `test_multi_person_detection.py` → Consolidar nos testes
- `test_phase1.py` → Consolidar nos testes
- `test_visualizations.py` → Consolidar nos testes  
- `validate_phase1.py` → Consolidar nos testes
- `visualize_all_detections.py` → Consolidar no tools/
- `visualize_detections.py` → Consolidar no tools/

### 2. Arquivos temporários e resultados
- `multi_person_detection_results.json` → Remover (resultado temporário)
- `haarcascade_frontalface_default.xml` → Mover para data/models/
- `__pycache__/` → Remover
- `visualizations/` → Mover para data/quality/visualizations/

### 3. Requirements
- `requirements_phase1_complete.txt` → Remover (duplicado)
- Manter apenas `requirements.txt`

### 4. Relatórios temporários (manter na docs/)
- `PHASE1_FINAL_IMPLEMENTATION_REPORT.md` → Mover para docs/
- `PHASE1_IMPLEMENTATION_SUMMARY.md` → Mover para docs/

### 5. Scripts duplicados no src/core/
- `person_detector_simplified.py` → Remover (version simplificada)
- Manter apenas `person_detector.py`

## Scripts consolidados para MANTER no tools/:

### A. Para análise e debug:
- `analysis_tools.py` (consolidado)
- `debug_tools.py` (consolidado)  

### B. Para testes e validação:
- `validation_tools.py` (consolidado)
- `testing_suite.py` (consolidado)

### C. Para visualização:
- `visualization_tools.py` (consolidado)

### D. Utilitários existentes (manter):
- `ai_prediction_tester.py`
- `demo_system.py`
- `health_check_complete.py`
- `integration_test.py`
- `quality_analyzer.py`

## Estrutura final limpa:

```
Photo-Culling/
├── main.py
├── config.json
├── requirements.txt
├── README.md
├── src/
│   ├── core/
│   │   ├── feature_extractor.py
│   │   ├── person_detector.py (único)
│   │   ├── exposure_analyzer.py
│   │   └── outros módulos core
│   ├── utils/
│   └── web/
├── tools/
│   ├── analysis_tools.py (consolidado)
│   ├── debug_tools.py (consolidado)
│   ├── validation_tools.py (consolidado)
│   ├── testing_suite.py (consolidado)
│   ├── visualization_tools.py (consolidado)
│   └── módulos existentes...
├── data/
│   ├── input/
│   ├── features/
│   ├── labels/
│   ├── models/
│   │   └── haarcascade_frontalface_default.xml
│   └── quality/
│       └── visualizations/
└── docs/
    ├── PHASE1_FINAL_IMPLEMENTATION_REPORT.md
    ├── PHASE1_IMPLEMENTATION_SUMMARY.md
    └── outros docs...
```
