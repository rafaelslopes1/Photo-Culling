#!/usr/bin/env python3
"""
Consolidated AI Classifier for Image Quality Assessment
Sistema de IA consolidado para classificação de qualidade de imagens
Combina funcionalidades de treinamento, análise e predição
"""

import numpy as np
import pandas as pd
import sqlite3
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIClassifier:
    """
    Sistema consolidado de IA para classificação de qualidade de imagens
    Combina treinamento, análise de performance e predição em um sistema unificado
    """
    
    def __init__(self, labels_db="data/labels/labels.db", 
                 features_db="data/features/features.db",
                 models_dir="data/models"):
        
        self.labels_db = labels_db
        self.features_db = features_db
        self.models_dir = models_dir
        
        # Create directories
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(os.path.dirname(labels_db), exist_ok=True)
        os.makedirs(os.path.dirname(features_db), exist_ok=True)
        
        # Initialize models and scalers
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.class_labels = {}
        self.best_model_name = None
        
        logger.info("🤖 AI Classifier initialized")
    
    def load_training_data(self):
        """
        Carrega dados de treinamento combinando labels e features
        
        Returns:
            pandas.DataFrame: Dataset combinado para treinamento
        """
        logger.info("📊 Loading training data...")
        
        # Load labels
        labels_df = self._load_labels()
        if labels_df.empty:
            raise ValueError("Nenhum dado rotulado encontrado. Rotule algumas imagens primeiro.")
        
        logger.info(f"✓ Carregados {len(labels_df)} rótulos")
        
        # Load features
        features_df = self._load_features()
        if features_df.empty:
            raise ValueError("Nenhuma feature encontrada. Execute a extração de features primeiro.")
        
        logger.info(f"✓ Carregadas features para {len(features_df)} imagens")
        
        # Merge labels and features
        combined_df = labels_df.merge(features_df, on='filename', how='inner')
        
        if combined_df.empty:
            raise ValueError("Nenhuma correspondência entre rótulos e features.")
        
        logger.info(f"✓ Dataset combinado: {len(combined_df)} imagens")
        
        # Analyze dataset balance
        self._analyze_dataset_balance(combined_df)
        
        return combined_df
    
    def train_models(self, data=None, test_size=0.2, cv_folds=3):
        """
        Treina múltiplos modelos e compara performance
        
        Args:
            data: DataFrame com dados de treinamento (opcional)
            test_size: Proporção para teste
            cv_folds: Número de folds para validação cruzada
            
        Returns:
            dict: Resultados de performance dos modelos
        """
        if data is None:
            data = self.load_training_data()
        
        # Prepare data
        X, y = self._prepare_features_and_targets(data)
        
        # Define models to train
        models_config = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class='ovr'
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        }
        
        results = {}
        
        # Train each model
        for name, model in models_config.items():
            logger.info(f"🏋️ Treinando {name}...")
            
            try:
                # Create pipeline with scaling
                pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('classifier', model)
                ])
                
                # Cross-validation
                cv_scores = cross_val_score(
                    pipeline, X, y, 
                    cv=cv_folds, 
                    scoring='accuracy',
                    n_jobs=-1
                )
                
                # Train on full data for final model
                pipeline.fit(X, y)
                
                # Store results
                results[name] = {
                    'model': pipeline,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores.tolist()
                }
                
                # Store model and scaler
                self.models[name] = pipeline
                
                logger.info(f"✓ {name}: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
                
            except Exception as e:
                logger.error(f"❌ Erro treinando {name}: {e}")
                continue
        
        # Find best model
        if results:
            self.best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
            logger.info(f"🏆 Melhor modelo: {self.best_model_name}")
            
            # Save best model
            self._save_best_model()
        
        return results
    
    def predict_batch(self, image_paths_or_features, return_probabilities=False):
        """
        Faz predições em lote para múltiplas imagens
        
        Args:
            image_paths_or_features: Lista de caminhos ou DataFrame com features
            return_probabilities: Se retornar probabilidades
            
        Returns:
            list: Predições ou tuplas (predição, probabilidades)
        """
        if self.best_model_name is None:
            raise ValueError("Nenhum modelo treinado encontrado. Treine um modelo primeiro.")
        
        best_model = self.models[self.best_model_name]
        
        # Handle different input types
        if isinstance(image_paths_or_features, (list, tuple)):
            # Extract features from image paths
            from .feature_extractor import FeatureExtractor
            extractor = FeatureExtractor(self.features_db)
            features_list = extractor.extract_batch(image_paths_or_features)
            
            if not features_list:
                return []
            
            # Convert to DataFrame
            features_df = pd.DataFrame(features_list)
            X = self._prepare_features_for_prediction(features_df)
            
        elif isinstance(image_paths_or_features, pd.DataFrame):
            X = self._prepare_features_for_prediction(image_paths_or_features)
        else:
            raise ValueError("Input deve ser lista de caminhos ou DataFrame")
        
        # Make predictions
        predictions = best_model.predict(X)
        
        if return_probabilities:
            probabilities = best_model.predict_proba(X)
            return list(zip(predictions, probabilities))
        
        return predictions.tolist()
    
    def analyze_model_performance(self, model_name=None):
        """
        Analisa performance detalhada do modelo
        
        Args:
            model_name: Nome do modelo (usa o melhor se None)
            
        Returns:
            dict: Análise de performance detalhada
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} não encontrado")
        
        # Load training data
        data = self.load_training_data()
        X, y = self._prepare_features_and_targets(data)
        
        model = self.models[model_name]
        
        # Make predictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y, y_pred)
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importance_scores = model.named_steps['classifier'].feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
        
        analysis = {
            'model_name': model_name,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'feature_importance': feature_importance.to_dict('records') if feature_importance is not None else None,
            'class_distribution': pd.Series(y).value_counts().to_dict(),
            'prediction_confidence': {
                'mean': np.mean(np.max(y_proba, axis=1)),
                'std': np.std(np.max(y_proba, axis=1)),
                'min': np.min(np.max(y_proba, axis=1)),
                'max': np.max(np.max(y_proba, axis=1))
            }
        }
        
        return analysis
    
    def get_prediction_for_image(self, image_path):
        """
        Faz predição para uma única imagem
        
        Args:
            image_path: Caminho da imagem
            
        Returns:
            dict: Predição com probabilidades e confiança
        """
        predictions = self.predict_batch([image_path], return_probabilities=True)
        
        if not predictions:
            return None
        
        prediction, probabilities = predictions[0]
        
        # Get class names
        classes = self.models[self.best_model_name].classes_
        
        # Create probability dict
        prob_dict = {cls: prob for cls, prob in zip(classes, probabilities[0])}
        
        # Format prediction
        formatted_prediction = self._format_prediction_label(prediction)
        
        return {
            'prediction': prediction,
            'formatted_prediction': formatted_prediction,
            'confidence': np.max(probabilities[0]),
            'probabilities': prob_dict,
            'model_used': self.best_model_name
        }
    
    def _load_labels(self):
        """Carrega rótulos do banco de dados"""
        try:
            conn = sqlite3.connect(self.labels_db)
            
            query = """
                SELECT filename, label_type, score, rejection_reason, timestamp
                FROM labels
                ORDER BY timestamp DESC
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                return pd.DataFrame()
            
            # Create target column
            df['target'] = df.apply(self._create_target_label, axis=1)
            
            return df[['filename', 'target', 'label_type', 'score', 'rejection_reason']]
            
        except Exception as e:
            logger.error(f"Erro carregando labels: {e}")
            return pd.DataFrame()
    
    def _load_features(self):
        """Carrega features do banco de dados"""
        try:
            conn = sqlite3.connect(self.features_db)
            df = pd.read_sql_query("SELECT * FROM image_features", conn)
            conn.close()
            
            # Remove non-numeric columns
            non_numeric_cols = [
                'format', 'camera_make', 'camera_model', 'f_number', 
                'focal_length', 'datetime_taken', 'dominant_colors', 
                'face_areas', 'uniqueness_hash', 'extraction_timestamp', 
                'extraction_version'
            ]
            
            for col in non_numeric_cols:
                if col in df.columns:
                    df = df.drop(col, axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Erro carregando features: {e}")
            return pd.DataFrame()
    
    def _create_target_label(self, row):
        """Cria label de destino baseado no tipo de rótulo"""
        if row['label_type'] == 'quality':
            return f'quality_{int(row["score"])}'
        else:
            return f'reject_{row["rejection_reason"]}'
    
    def _prepare_features_and_targets(self, data):
        """Prepara features e targets para treinamento"""
        # Feature columns (exclude metadata)
        feature_cols = [col for col in data.columns 
                       if col not in ['filename', 'target', 'label_type', 'score', 'rejection_reason']]
        
        self.feature_columns = feature_cols
        
        X = data[feature_cols].copy()
        y = data['target'].copy()
        
        # Fill NaN values
        X = X.fillna(0)
        
        return X, y
    
    def _prepare_features_for_prediction(self, features_df):
        """Prepara features para predição"""
        # Ensure we have the same columns as training
        missing_cols = set(self.feature_columns) - set(features_df.columns)
        extra_cols = set(features_df.columns) - set(self.feature_columns) - {'filename'}
        
        # Add missing columns with zeros
        for col in missing_cols:
            features_df[col] = 0
        
        # Remove extra columns
        for col in extra_cols:
            if col in features_df.columns:
                features_df = features_df.drop(col, axis=1)
        
        # Select and order columns
        X = features_df[self.feature_columns].copy()
        X = X.fillna(0)
        
        return X
    
    def _analyze_dataset_balance(self, data):
        """Analisa balanceamento do dataset"""
        target_counts = data['target'].value_counts()
        total = len(data)
        
        logger.info("\n📈 Análise do Dataset:")
        logger.info("=" * 50)
        
        logger.info("Distribuição das Classes:")
        for target, count in target_counts.items():
            percentage = (count / total) * 100
            logger.info(f"  {target}: {count} ({percentage:.1f}%)")
        
        # Check balance
        max_count = target_counts.max()
        min_count = target_counts.min()
        balance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        logger.info(f"\nTotal de amostras: {total}")
        logger.info(f"Número de classes: {len(target_counts)}")
        
        if balance_ratio > 3:
            logger.warning(f"⚠️  Dataset desbalanceado (ratio: {balance_ratio:.1f}:1)")
            logger.warning("   Considere coletar mais dados das classes minoritárias")
        else:
            logger.info(f"✓ Dataset relativamente balanceado (ratio: {balance_ratio:.1f}:1)")
    
    def _format_prediction_label(self, prediction):
        """Formata label de predição para exibição"""
        if prediction.startswith('quality_'):
            score = prediction.split('_')[1]
            stars = '⭐' * int(score)
            return f"Qualidade {score} {stars}"
        elif prediction.startswith('reject_'):
            reason = prediction.split('_')[1]
            reason_map = {
                'blur': '💫 Desfocada',
                'dark': '🌑 Muito escura',
                'light': '☀️ Muito clara',
                'cropped': '✂️ Cortada',
                'other': '❓ Outros'
            }
            return reason_map.get(reason, f"Rejeitada ({reason})")
        
        return prediction
    
    def _save_best_model(self):
        """Salva o melhor modelo"""
        if self.best_model_name and self.best_model_name in self.models:
            model_path = os.path.join(self.models_dir, "best_model.joblib")
            
            model_data = {
                'model': self.models[self.best_model_name],
                'model_name': self.best_model_name,
                'feature_columns': self.feature_columns,
                'trained_at': datetime.now().isoformat(),
                'version': '2.0_consolidated'
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"✓ Modelo salvo em {model_path}")
    
    def load_best_model(self):
        """Carrega o melhor modelo salvo"""
        model_path = os.path.join(self.models_dir, "best_model.joblib")
        
        if not os.path.exists(model_path):
            logger.warning("Nenhum modelo salvo encontrado")
            return False
        
        try:
            model_data = joblib.load(model_path)
            
            # Verificar se é um dicionário com estrutura esperada ou um modelo direto
            if isinstance(model_data, dict):
                # Estrutura nova (dicionário)
                if 'model' not in model_data:
                    logger.error("Modelo salvo não tem estrutura válida (falta 'model')")
                    return False
                
                if 'model_name' not in model_data:
                    logger.error("Modelo salvo não tem estrutura válida (falta 'model_name')")
                    return False
                    
                if 'feature_columns' not in model_data:
                    logger.error("Modelo salvo não tem estrutura válida (falta 'feature_columns')")
                    return False
                
                self.models[model_data['model_name']] = model_data['model']
                self.best_model_name = model_data['model_name']
                self.feature_columns = model_data['feature_columns']
                
            else:
                # Estrutura antiga (modelo direto)
                logger.warning("Carregando modelo com estrutura antiga")
                self.models['legacy_model'] = model_data
                self.best_model_name = 'legacy_model'
                # Tentar carregar metadados separadamente
                metadata_path = os.path.join(self.models_dir, "best_model_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self.feature_columns = metadata.get('feature_columns', [])
                else:
                    logger.warning("Metadados do modelo não encontrados")
                    self.feature_columns = []
            
            logger.info(f"✓ Modelo carregado: {self.best_model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Erro carregando modelo: {e}")
            return False

# Convenience functions
def train_classifier_from_folder(input_folder="data/input", 
                                labels_db="data/labels/labels.db",
                                features_db="data/features/features.db"):
    """
    Função de conveniência para treinar classificador completo
    
    Args:
        input_folder: Pasta com imagens
        labels_db: Banco de dados de rótulos
        features_db: Banco de dados de features
        
    Returns:
        AIClassifier: Classificador treinado
    """
    # Extract features if needed
    if not os.path.exists(features_db):
        from .feature_extractor import extract_features_from_folder
        logger.info("Extraindo features das imagens...")
        extract_features_from_folder(input_folder, features_db)
    
    # Initialize and train classifier
    classifier = AIClassifier(labels_db, features_db)
    
    try:
        results = classifier.train_models()
        logger.info("✅ Treinamento concluído com sucesso!")
        return classifier, results
    except Exception as e:
        logger.error(f"❌ Erro no treinamento: {e}")
        return classifier, None

if __name__ == "__main__":
    # Example usage
    classifier = AIClassifier()
    
    try:
        results = classifier.train_models()
        print("🎉 Treinamento concluído!")
        
        # Analyze best model
        analysis = classifier.analyze_model_performance()
        print(f"Acurácia: {analysis['accuracy']:.1%}")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        print("Certifique-se de ter imagens rotuladas e features extraídas.")
