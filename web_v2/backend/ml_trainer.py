#!/usr/bin/env python3
"""
ML Training Pipeline for Expert-Trained Photo Quality Assessment
Pipeline de treinamento de modelos usando avaliações de especialistas
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import json
import sqlite3
import logging
from pathlib import Path
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'src'))

from src.core.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

class ExpertTrainedMLPipeline:
    """
    Pipeline de Machine Learning treinado com avaliações de especialistas
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(project_root / 'web_v2' / 'backend' / 'expert_evaluations.db')
        self.feature_extractor = FeatureExtractor()
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
        # Define feature columns
        self.technical_features = [
            'sharpness_laplacian', 'sharpness_sobel', 'sharpness_fft',
            'brightness_mean', 'brightness_std', 'contrast_rms',
            'saturation_mean', 'noise_level', 'color_variance',
            'face_count', 'face_areas_mean', 'skin_ratio',
            'rule_of_thirds_score', 'symmetry_score', 'edge_density',
            'texture_complexity', 'exposure_quality_score'
        ]
        
    def load_expert_data(self) -> pd.DataFrame:
        """
        Carrega dados de avaliações de especialistas do banco
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
            SELECT 
                image_filename,
                evaluator_id,
                overall_quality,
                global_sharpness,
                person_sharpness,
                exposure_quality,
                composition_quality,
                emotional_impact,
                technical_execution,
                approve_for_portfolio,
                approve_for_client,
                approve_for_social,
                needs_editing,
                complete_reject,
                confidence_level,
                evaluation_time_seconds
            FROM expert_evaluation
            WHERE overall_quality IS NOT NULL
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            logger.info(f"Loaded {len(df)} expert evaluations")
            return df
            
        except Exception as e:
            logger.error(f"Error loading expert data: {e}")
            return pd.DataFrame()
    
    def load_technical_features(self, image_filenames: List[str]) -> pd.DataFrame:
        """
        Carrega features técnicas para as imagens avaliadas
        """
        features_data = []
        
        for filename in image_filenames:
            try:
                # Try to load from existing feature database first
                feature_db_path = project_root / 'data' / 'features' / 'features.db'
                
                if feature_db_path.exists():
                    conn = sqlite3.connect(str(feature_db_path))
                    query = "SELECT * FROM image_features WHERE filename = ?"
                    existing_features = pd.read_sql_query(query, conn, params=(filename,))
                    conn.close()
                    
                    if not existing_features.empty:
                        features_dict = existing_features.iloc[0].to_dict()
                        features_dict['filename'] = filename
                        features_data.append(features_dict)
                        continue
                
                # Extract features if not in database
                image_path = project_root / 'data' / 'input' / filename
                if image_path.exists():
                    features = self.feature_extractor.extract_features(str(image_path))
                    features['filename'] = filename
                    features_data.append(features)
                else:
                    logger.warning(f"Image not found: {filename}")
                    
            except Exception as e:
                logger.error(f"Error extracting features for {filename}: {e}")
        
        if features_data:
            df = pd.DataFrame(features_data)
            logger.info(f"Loaded technical features for {len(df)} images")
            return df
        else:
            return pd.DataFrame()
    
    def merge_expert_technical_data(self) -> pd.DataFrame:
        """
        Combina dados de especialistas com features técnicas
        """
        expert_df = self.load_expert_data()
        if expert_df.empty:
            logger.error("No expert data available")
            return pd.DataFrame()
        
        unique_images = expert_df['image_filename'].unique()
        technical_df = self.load_technical_features(unique_images)
        
        if technical_df.empty:
            logger.error("No technical features available")
            return pd.DataFrame()
        
        # Merge datasets
        merged_df = expert_df.merge(
            technical_df, 
            left_on='image_filename', 
            right_on='filename', 
            how='inner'
        )
        
        logger.info(f"Merged dataset has {len(merged_df)} samples")
        return merged_df
    
    def prepare_training_data(self, merged_df: pd.DataFrame) -> Dict:
        """
        Prepara dados para treinamento
        """
        # Select feature columns that exist in the data
        available_features = [col for col in self.technical_features if col in merged_df.columns]
        
        if not available_features:
            logger.error("No technical features available for training")
            return {}
        
        self.feature_columns = available_features
        
        # Prepare feature matrix
        X = merged_df[self.feature_columns].fillna(0)
        
        # Prepare different target variables
        targets = {
            'overall_quality': merged_df['overall_quality'],
            'global_sharpness': merged_df['global_sharpness'],
            'person_sharpness': merged_df['person_sharpness'],
            'exposure_quality': merged_df['exposure_quality'],
            'composition_quality': merged_df['composition_quality'],
            'portfolio_approval': merged_df['approve_for_portfolio'].astype(int),
            'client_approval': merged_df['approve_for_client'].astype(int),
            'needs_editing': merged_df['needs_editing'].astype(int),
            'complete_reject': merged_df['complete_reject'].astype(int)
        }
        
        # Remove rows with missing targets
        valid_samples = {}
        for target_name, target_values in targets.items():
            mask = target_values.notna()
            if mask.sum() > 0:
                valid_samples[target_name] = {
                    'X': X[mask],
                    'y': target_values[mask],
                    'sample_count': mask.sum()
                }
        
        logger.info(f"Prepared training data with {len(self.feature_columns)} features")
        for target, data in valid_samples.items():
            logger.info(f"  {target}: {data['sample_count']} samples")
        
        return valid_samples
    
    def train_models(self, training_data: Dict) -> Dict:
        """
        Treina modelos para diferentes aspectos da qualidade
        """
        trained_models = {}
        
        for target_name, data in training_data.items():
            if data['sample_count'] < 10:
                logger.warning(f"Insufficient data for {target_name}: {data['sample_count']} samples")
                continue
            
            X, y = data['X'], data['y']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Choose model type based on target
            if target_name in ['portfolio_approval', 'client_approval', 'needs_editing', 'complete_reject']:
                # Classification
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10,
                    min_samples_split=5
                )
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                logger.info(f"Trained {target_name} classifier - Accuracy: {accuracy:.3f}")
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                logger.info(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                
            else:
                # Regression
                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10,
                    min_samples_split=5
                )
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                logger.info(f"Trained {target_name} regressor - RMSE: {rmse:.3f}")
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
                cv_rmse = np.sqrt(-cv_scores.mean())
                logger.info(f"  CV RMSE: {cv_rmse:.3f} (+/- {np.sqrt(cv_scores.std() * 2):.3f})")
            
            # Store model and scaler
            trained_models[target_name] = {
                'model': model,
                'scaler': scaler,
                'feature_columns': self.feature_columns,
                'model_type': 'classifier' if target_name in ['portfolio_approval', 'client_approval', 'needs_editing', 'complete_reject'] else 'regressor',
                'training_samples': len(X_train),
                'test_performance': accuracy if target_name in ['portfolio_approval', 'client_approval', 'needs_editing', 'complete_reject'] else rmse
            }
        
        return trained_models
    
    def save_models(self, models: Dict, model_dir: str = None):
        """
        Salva modelos treinados
        """
        if not model_dir:
            model_dir = project_root / 'data' / 'models' / 'expert_trained'
        
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, model_data in models.items():
            # Save model
            model_path = model_dir / f"{model_name}_model.joblib"
            joblib.dump(model_data['model'], model_path)
            
            # Save scaler
            scaler_path = model_dir / f"{model_name}_scaler.joblib"
            joblib.dump(model_data['scaler'], scaler_path)
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'model_type': model_data['model_type'],
                'feature_columns': model_data['feature_columns'],
                'training_samples': model_data['training_samples'],
                'test_performance': model_data['test_performance'],
                'trained_at': datetime.now().isoformat(),
                'model_path': str(model_path),
                'scaler_path': str(scaler_path)
            }
            
            metadata_path = model_dir / f"{model_name}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved {model_name} model to {model_path}")
        
        # Save master metadata
        master_metadata = {
            'training_timestamp': datetime.now().isoformat(),
            'total_models': len(models),
            'models': {name: data['model_type'] for name, data in models.items()},
            'feature_columns': self.feature_columns
        }
        
        master_path = model_dir / 'training_session_metadata.json'
        with open(master_path, 'w') as f:
            json.dump(master_metadata, f, indent=2)
        
        logger.info(f"Saved master metadata to {master_path}")
    
    def load_models(self, model_dir: str = None) -> Dict:
        """
        Carrega modelos treinados
        """
        if not model_dir:
            model_dir = project_root / 'data' / 'models' / 'expert_trained'
        
        model_dir = Path(model_dir)
        if not model_dir.exists():
            logger.error(f"Model directory not found: {model_dir}")
            return {}
        
        loaded_models = {}
        
        # Find all metadata files
        for metadata_file in model_dir.glob("*_metadata.json"):
            if metadata_file.name == 'training_session_metadata.json':
                continue
            
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                model_name = metadata['model_name']
                
                # Load model and scaler
                model = joblib.load(metadata['model_path'])
                scaler = joblib.load(metadata['scaler_path'])
                
                loaded_models[model_name] = {
                    'model': model,
                    'scaler': scaler,
                    'metadata': metadata
                }
                
                logger.info(f"Loaded {model_name} model")
                
            except Exception as e:
                logger.error(f"Error loading model from {metadata_file}: {e}")
        
        return loaded_models
    
    def predict_image_quality(self, image_path: str, models: Dict = None) -> Dict:
        """
        Prediz qualidade de uma imagem usando modelos treinados
        """
        if models is None:
            models = self.load_models()
        
        if not models:
            logger.error("No models available for prediction")
            return {}
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(image_path)
            
            # Prepare feature vector
            feature_vector = []
            for col in self.feature_columns:
                feature_vector.append(features.get(col, 0))
            
            X = np.array(feature_vector).reshape(1, -1)
            
            # Make predictions
            predictions = {}
            
            for model_name, model_data in models.items():
                try:
                    # Scale features
                    X_scaled = model_data['scaler'].transform(X)
                    
                    # Predict
                    if model_data['metadata']['model_type'] == 'classifier':
                        prediction = model_data['model'].predict(X_scaled)[0]
                        probabilities = model_data['model'].predict_proba(X_scaled)[0]
                        predictions[model_name] = {
                            'prediction': bool(prediction),
                            'confidence': float(max(probabilities))
                        }
                    else:
                        prediction = model_data['model'].predict(X_scaled)[0]
                        predictions[model_name] = {
                            'prediction': float(prediction),
                            'rating': min(5, max(1, round(prediction)))  # Convert to 1-5 scale
                        }
                        
                except Exception as e:
                    logger.error(f"Error predicting with {model_name}: {e}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting image quality: {e}")
            return {}
    
    def run_full_training_pipeline(self):
        """
        Executa pipeline completo de treinamento
        """
        logger.info("Starting expert-trained ML pipeline")
        
        # Load and merge data
        merged_df = self.merge_expert_technical_data()
        if merged_df.empty:
            logger.error("No data available for training")
            return False
        
        # Prepare training data
        training_data = self.prepare_training_data(merged_df)
        if not training_data:
            logger.error("Could not prepare training data")
            return False
        
        # Train models
        models = self.train_models(training_data)
        if not models:
            logger.error("No models were trained")
            return False
        
        # Save models
        self.save_models(models)
        
        logger.info("Expert-trained ML pipeline completed successfully")
        return True

def main():
    """
    Função principal para executar treinamento
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    pipeline = ExpertTrainedMLPipeline()
    success = pipeline.run_full_training_pipeline()
    
    if success:
        print("✅ Treinamento concluído com sucesso!")
        print("📊 Modelos salvos em: data/models/expert_trained/")
        print("🧠 Sistema pronto para predições baseadas em expertise!")
    else:
        print("❌ Falha no treinamento. Verifique os logs para detalhes.")

if __name__ == "__main__":
    main()
