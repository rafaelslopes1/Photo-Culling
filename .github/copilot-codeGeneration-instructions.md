# 🤖 GitHub Copilot Instructions - Photo Culling System

## 📋 Project Overview

This is a **Photo Culling System v2.0** - an intelligent photo classification and curation system with optimized blur detection and AI-powered quality assessment. The system combines automated processing with manual labeling capabilities through a web interface.

## 🎯 Core Development Guidelines

### Language and Communication Rules
- **Code Language**: Always write code in American English (comments, variable names, function names, etc.)
- **Error Messages**: Write error messages in Portuguese (pt-BR)
- **User Communication**: When responding to user questions, always use Portuguese (pt-BR)
- **Documentation**: Use English for technical documentation, Portuguese for user-facing content

### Code Quality Standards
```python
# ✅ Good - English code with Portuguese error messages
def analyze_image_quality(image_path: str) -> Dict:
    """
    Analyze image quality using blur detection and exposure analysis
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError("Imagem não encontrada no caminho especificado")
        
        # Process image analysis
        return process_analysis(image_path)
    except Exception as e:
        logger.error(f"Erro ao analisar qualidade da imagem: {e}")
        raise
```

## 🏗️ Project Architecture

### Directory Structure
```
Photo-Culling/
├── src/                     # Source code
│   ├── core/               # Core functionality
│   │   ├── ai_classifier.py
│   │   ├── feature_extractor.py
│   │   ├── image_processor.py
│   │   └── image_quality_analyzer.py
│   ├── web/                # Web interface
│   │   ├── app.py
│   │   └── templates/
│   └── utils/              # Utilities
├── data/                   # Data storage
│   ├── input/              # Input images
│   ├── features/           # Feature database
│   ├── labels/             # Labels database
│   ├── models/             # AI models
│   └── quality/            # Quality analysis
├── docs/                   # Documentation
├── tools/                  # Utility scripts
├── config.json             # Configuration
├── requirements.txt        # Python dependencies
└── main.py                # Main entry point
```

### Core Components Understanding
1. **FeatureExtractor**: Extracts technical and visual features from images
2. **AIClassifier**: Machine learning models for quality prediction  
3. **ImageProcessor**: Main processing pipeline with blur detection
4. **WebLabelingApp**: Flask-based manual labeling interface
5. **ImageQualityAnalyzer**: Specialized blur and quality analysis

## 🛠️ Development Tools and Commands

### Package Management
```bash
# ✅ Always use pip for Python dependencies
pip install package_name
pip install -r requirements.txt
pip freeze > requirements.txt

# ❌ Never manually edit requirements.txt
# Let pip manage dependencies automatically
```

### Git Workflow
```bash
# Daily workflow
git status                  # Check current state
git add .                  # Stage changes
git commit -m "feat: implement person detection analysis"
git push origin main       # Push to remote

# Branch management
git checkout -b feature/person-analysis
git merge main            # Keep branch updated
git rebase -i HEAD~3      # Clean commit history
```

### Git Commit Standards
Follow conventional commits:
```bash
# Feature additions
git commit -m "feat: add dominant person detection algorithm"

# Bug fixes  
git commit -m "fix: resolve blur threshold calculation issue"

# Documentation
git commit -m "docs: update API documentation for person analysis"

# Refactoring
git commit -m "refactor: optimize face detection performance"

# Configuration
git commit -m "config: update blur detection thresholds"

# Tests
git commit -m "test: add unit tests for exposure analysis"
```

## 📚 Key Libraries and Technologies

### Core Dependencies
```python
# Computer Vision
import cv2                 # OpenCV for image processing
import numpy as np         # Numerical operations
from PIL import Image      # Image handling

# Machine Learning
import pandas as pd        # Data manipulation
import sklearn            # ML algorithms
import joblib             # Model persistence

# Web Framework
from flask import Flask   # Web interface
import sqlite3           # Database

# Advanced Features (Optional)
import face_recognition   # Face detection and recognition
import mediapipe as mp   # Person detection and pose analysis
```

### Installation Commands
```bash
# Core requirements
pip install opencv-python pillow numpy pandas scikit-learn flask

# Optional advanced features
pip install face-recognition mediapipe ultralytics

# Development tools
pip install pytest black flake8 mypy
```

## 🔧 Configuration Management

### Main Configuration (config.json)
```json
{
  "processing_settings": {
    "blur_detection_optimized": {
      "enabled": true,
      "strategy": "balanced",
      "strategies": {
        "conservative": {"threshold": 50},
        "balanced": {"threshold": 78}, 
        "aggressive": {"threshold": 145}
      }
    },
    "person_analysis": {
      "enabled": false,
      "min_person_area_ratio": 0.05,
      "face_recognition_threshold": 0.6
    }
  }
}
```

### Environment Variables
```bash
# Development
export FLASK_ENV=development
export FLASK_DEBUG=1

# Production
export FLASK_ENV=production
export PHOTO_CULLING_DATA_DIR=/data/photos
```

## 🔄 Periodic Maintenance Tasks

### Daily Tasks
```bash
# 1. Check system health
python tools/health_check_complete.py

# 2. Run integration tests
python tools/integration_test.py

# 3. Check for lint issues
black src/ --check
flake8 src/

# 4. Commit any fixes
git add . && git commit -m "fix: daily maintenance updates"
```

### Weekly Tasks
```bash
# 1. Update dependencies
pip list --outdated
pip install --upgrade package_name

# 2. Run full test suite
python -m pytest tests/ -v

# 3. Clean up databases
python tools/cleanup_databases.py

# 4. Generate performance report
python tools/performance_analyzer.py
```

### Monthly Tasks
```bash
# 1. Full system backup
python tools/backup_system.py

# 2. Refactor code quality
black src/
isort src/

# 3. Update documentation
# Review and update README.md, docs/

# 4. Performance optimization review
python tools/profile_performance.py
```

## 📋 Roadmap Consultation

### Current Roadmap Location
- **Main Roadmap**: `docs/PHOTO_SELECTION_REFINEMENT_PROMPT.md`
- **Implementation Status**: `docs/INTEGRATION_STATUS_FINAL.md`
- **Blur Analysis**: `docs/BLUR_ANALYSIS_EXECUTIVE_SUMMARY.md`

### Implementation Phases
```python
# Check current phase
def get_current_phase():
    """
    Phase 1: Basic blur detection (✅ Complete)
    Phase 2: Person detection (🔄 In Progress)
    Phase 3: Face recognition (⏳ Planned)
    Phase 4: Advanced analysis (⏳ Planned)
    """
    return check_implementation_status()
```

### Feature Priority Matrix
1. **High Priority**: Blur detection optimization, person detection
2. **Medium Priority**: Face recognition, advanced composition analysis  
3. **Low Priority**: Aesthetic scoring, advanced ML models

## 🧹 Code Cleanup and Refactoring

### Code Quality Checklist
```python
# ✅ Functions should be focused and single-purpose
def calculate_blur_score(image: np.ndarray) -> float:
    """Calculate blur score using Variance of Laplacian method"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# ✅ Use type hints
def process_image_batch(images: List[str]) -> Dict[str, float]:
    pass

# ✅ Handle errors gracefully
try:
    result = process_image(image_path)
except Exception as e:
    logger.error(f"Erro no processamento da imagem: {e}")
    return default_result()
```

### Refactoring Guidelines
```bash
# Remove dead code
grep -r "TODO\|FIXME\|XXX" src/

# Optimize imports
isort src/ --profile black

# Format code
black src/ --line-length 88

# Check type hints
mypy src/ --ignore-missing-imports
```

### Performance Optimization
```python
# ✅ Use vectorized operations
blur_scores = np.array([cv2.Laplacian(img, cv2.CV_64F).var() for img in images])

# ✅ Cache expensive operations
@lru_cache(maxsize=128)
def load_ai_model(model_path: str):
    return joblib.load(model_path)

# ✅ Use generators for large datasets
def process_images_generator(image_dir: str):
    for image_path in Path(image_dir).glob("*.jpg"):
        yield process_single_image(image_path)
```

## 📊 Database Management

### SQLite Database Structure
```sql
-- Features database
CREATE TABLE image_features (
    filename TEXT PRIMARY KEY,
    sharpness_laplacian REAL,
    brightness_mean REAL,
    face_count INTEGER,
    -- Person analysis (new)
    dominant_person_score REAL,
    person_count INTEGER,
    face_encodings TEXT
);

-- Labels database  
CREATE TABLE labels (
    filename TEXT PRIMARY KEY,
    label_type TEXT,
    score INTEGER,
    rejection_reason TEXT,
    timestamp TEXT
);
```

### Database Maintenance
```python
# Clean up orphaned records
def cleanup_database():
    """Remove records for non-existent images"""
    conn = sqlite3.connect('data/features/features.db')
    cursor = conn.cursor()
    
    cursor.execute("""
        DELETE FROM image_features 
        WHERE filename NOT IN (
            SELECT name FROM image_files
        )
    """)
    
    conn.commit()
    conn.close()
```

## 🚀 Development Workflow

### Feature Development Process
1. **Create Feature Branch**: `git checkout -b feature/new-feature`
2. **Implement Core Logic**: Start with `src/core/` modules
3. **Add Tests**: Create corresponding test files
4. **Update Configuration**: Modify `config.json` if needed
5. **Test Integration**: Run `python tools/integration_test.py`
6. **Update Documentation**: Add to relevant docs
7. **Commit and Push**: Follow commit standards
8. **Create Pull Request**: For review

### Testing Strategy
```python
# Unit tests
def test_blur_detection():
    image = load_test_image("blurry_sample.jpg")
    score = calculate_blur_score(image)
    assert score < 50, "Image should be detected as blurry"

# Integration tests
def test_full_pipeline():
    result = process_image_complete("test_image.jpg")
    assert 'blur_score' in result
    assert 'quality_rating' in result
```

### Error Handling Best Practices
```python
# ✅ Specific error handling
try:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
except cv2.error as e:
    logger.error(f"Erro do OpenCV: {e}")
    raise
except Exception as e:
    logger.error(f"Erro inesperado no processamento: {e}")
    raise
```

## 📈 Performance Monitoring

### Key Metrics to Track
```python
# Processing speed
images_per_second = total_images / processing_time

# Memory usage
memory_usage = psutil.Process().memory_info().rss / 1024 / 1024

# Accuracy metrics
accuracy = correct_predictions / total_predictions
```

### Performance Optimization Targets
- **Image Processing**: > 10 images/second
- **Memory Usage**: < 2GB for 1000 images
- **Blur Detection Accuracy**: > 90%
- **AI Classification Accuracy**: > 85%

## 🔍 Debugging and Troubleshooting

### Common Issues and Solutions
```python
# Issue: OpenCV not loading images
# Solution: Check file path and permissions
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {image_path}")

# Issue: SQLite database locked
# Solution: Ensure proper connection handling
try:
    conn = sqlite3.connect(db_path, timeout=10)
    # ... operations
finally:
    conn.close()

# Issue: Memory overflow with large batches
# Solution: Process in smaller chunks
def process_large_batch(images, chunk_size=100):
    for i in range(0, len(images), chunk_size):
        chunk = images[i:i + chunk_size]
        yield process_image_chunk(chunk)
```

### Debug Mode Configuration
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Add debug flags to config
DEBUG_CONFIG = {
    "debug_mode": True,
    "save_intermediate_results": True,
    "verbose_logging": True
}
```

## 🎯 Goals and Success Metrics

### Short-term Goals (1-2 weeks)
- [ ] Implement person detection pipeline
- [ ] Add exposure analysis functionality  
- [ ] Create person-focused quality assessment
- [ ] Update web interface with new categories

### Medium-term Goals (1-2 months)
- [ ] Implement face recognition and clustering
- [ ] Add advanced composition analysis
- [ ] Optimize processing performance
- [ ] Create comprehensive test suite

### Long-term Goals (3-6 months)
- [ ] Deploy production-ready system
- [ ] Implement advanced AI models
- [ ] Create mobile/desktop applications
- [ ] Establish continuous integration pipeline

### Success Metrics
- **Accuracy**: > 90% blur detection accuracy
- **Performance**: < 1 second per image processing
- **Usability**: < 5 minutes training time for new users
- **Reliability**: 99.9% uptime for web interface

---

## 🤝 Collaboration Guidelines

When working on this project:

1. **Always check existing code** before implementing new features
2. **Follow the established patterns** in the codebase
3. **Update documentation** alongside code changes
4. **Test thoroughly** before committing
5. **Ask questions** when architecture decisions are unclear
6. **Maintain backwards compatibility** when possible
7. **Keep commits atomic** and well-described
8. **Review and update the roadmap** regularly

Remember: This is a production system used for photo curation. Quality and reliability are paramount.