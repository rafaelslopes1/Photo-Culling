#!/usr/bin/env python3
"""
AI Prediction Tester - Sistema de Teste para Predições do Modelo
Ferramenta para testar o modelo atual em imagens não rotuladas
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from pathlib import Path
import json
from datetime import datetime

class AIPredictionTester:
    def __init__(self, model_path="models/current_model.joblib"):
        self.model_path = model_path
        self.model = None
        self.feature_columns = []
        self.class_labels = {}
        
    def train_current_model(self):
        """Treina o modelo atual com os dados disponíveis"""
        print("🤖 Treinando modelo atual...")
        
        # Load clean data
        data = self._load_clean_data()
        if len(data) < 10:
            print("❌ Dados insuficientes para treinar modelo")
            return False
        
        # Prepare features
        feature_cols = [col for col in data.columns 
                       if col not in ['filename', 'label_type', 'score', 'rejection_reason', 'target']]
        
        X = data[feature_cols].fillna(0)
        y = data['target']
        
        self.feature_columns = feature_cols
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=50,  # Reduced to prevent overfitting
            max_depth=5,      # Limit depth
            min_samples_split=5,  # Prevent splitting on small samples
            random_state=42
        )
        self.model.fit(X, y)
        
        # Create label mapping
        unique_labels = sorted(y.unique())
        self.class_labels = {label: self._format_label(label) for label in unique_labels}
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'features': self.feature_columns, 
            'labels': self.class_labels,
            'trained_at': datetime.now().isoformat()
        }, self.model_path)
        
        print(f"✅ Modelo treinado e salvo em {self.model_path}")
        print(f"📊 Classes: {len(self.class_labels)}")
        print(f"🔧 Features: {len(self.feature_columns)}")
        
        return True
    
    def load_model(self):
        """Carrega modelo salvo"""
        if not os.path.exists(self.model_path):
            print("❌ Modelo não encontrado. Treinando novo modelo...")
            return self.train_current_model()
        
        try:
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.feature_columns = data['features']
            self.class_labels = data['labels']
            print(f"✅ Modelo carregado de {self.model_path}")
            return True
        except Exception as e:
            print(f"❌ Erro ao carregar modelo: {e}")
            return self.train_current_model()
    
    def predict_unlabeled_samples(self, limit=20):
        """Testa predições em amostras não rotuladas"""
        if not self.model:
            if not self.load_model():
                return
        
        print(f"🔮 Testando predições em imagens não rotuladas...")
        
        # Get unlabeled images with features
        unlabeled = self._get_unlabeled_with_features(limit)
        
        if len(unlabeled) == 0:
            print("❌ Nenhuma imagem não rotulada com features encontrada")
            return
        
        # Prepare features
        X = unlabeled[self.feature_columns].fillna(0)
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Show results
        print(f"\\n📊 PREDIÇÕES PARA {len(unlabeled)} IMAGENS:")
        print("-" * 60)
        
        for i, (filename, pred) in enumerate(zip(unlabeled['filename'], predictions)):
            prob_max = probabilities[i].max()
            formatted_label = self.class_labels.get(pred, pred)
            confidence = "🔥 Alta" if prob_max > 0.8 else "🔸 Média" if prob_max > 0.6 else "🔹 Baixa"
            
            print(f"{i+1:2d}. {filename}")
            print(f"    Predição: {formatted_label}")
            print(f"    Confiança: {confidence} ({prob_max:.1%})")
            print()
    
    def test_on_labeled_data(self):
        """Testa modelo em dados já rotulados para verificar acurácia"""
        if not self.model:
            if not self.load_model():
                return
        
        print("🎯 Testando modelo em dados rotulados...")
        
        # Load test data
        data = self._load_clean_data()
        
        # Prepare features
        X = data[self.feature_columns].fillna(0)
        y_true = data['target']
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate accuracy
        accuracy = (y_pred == y_true).mean()
        
        print(f"\\n📈 RESULTADO DO TESTE:")
        print(f"  Acurácia: {accuracy:.1%}")
        
        # Show some examples
        print(f"\\n🔍 EXEMPLOS DE PREDIÇÕES:")
        print("-" * 50)
        
        for i in range(min(10, len(data))):
            filename = data.iloc[i]['filename']
            true_label = self._format_label(y_true.iloc[i])
            pred_label = self._format_label(y_pred[i])
            correct = "✅" if y_true.iloc[i] == y_pred[i] else "❌"
            
            print(f"{correct} {filename}")
            print(f"    Real: {true_label}")
            print(f"    Pred: {pred_label}")
            print()
    
    def _load_clean_data(self):
        """Carrega dados limpos"""
        # Load labels
        labels_conn = sqlite3.connect('web_labeling/data/labels.db')
        labels_df = pd.read_sql_query('''
            SELECT filename, label_type, score, rejection_reason 
            FROM labels
        ''', labels_conn)
        labels_conn.close()
        
        # Load features  
        features_conn = sqlite3.connect('web_labeling/data/features.db')
        features_df = pd.read_sql_query('SELECT * FROM image_features', features_conn)
        features_conn.close()
        
        # Remove non-numeric columns
        non_numeric_cols = ['format', 'camera_make', 'camera_model', 'f_number', 
                           'focal_length', 'datetime_taken', 'dominant_colors', 
                           'face_areas', 'uniqueness_hash', 'extraction_timestamp', 
                           'extraction_version']
        
        for col in non_numeric_cols:
            if col in features_df.columns:
                features_df = features_df.drop(col, axis=1)
        
        # Create target
        labels_df['target'] = labels_df.apply(self._create_target, axis=1)
        
        # Merge
        combined = pd.merge(labels_df, features_df, on='filename', how='inner')
        return combined
    
    def _get_unlabeled_with_features(self, limit):
        """Obtém imagens não rotuladas que têm features"""
        # Load all features
        features_conn = sqlite3.connect('web_labeling/data/features.db')
        features_df = pd.read_sql_query('SELECT * FROM image_features', features_conn)
        features_conn.close()
        
        # Load labeled filenames
        labels_conn = sqlite3.connect('web_labeling/data/labels.db')
        labeled_df = pd.read_sql_query('SELECT DISTINCT filename FROM labels', labels_conn)
        labels_conn.close()
        
        # Filter unlabeled
        unlabeled = features_df[~features_df['filename'].isin(labeled_df['filename'])]
        
        # Remove non-numeric columns
        non_numeric_cols = ['format', 'camera_make', 'camera_model', 'f_number', 
                           'focal_length', 'datetime_taken', 'dominant_colors', 
                           'face_areas', 'uniqueness_hash', 'extraction_timestamp', 
                           'extraction_version']
        
        for col in non_numeric_cols:
            if col in unlabeled.columns:
                unlabeled = unlabeled.drop(col, axis=1)
        
        return unlabeled.head(limit)
    
    def _create_target(self, row):
        """Cria target label"""
        if row['label_type'] == 'quality':
            return f'quality_{int(row["score"])}'
        else:
            return f'reject_{row["rejection_reason"]}'
    
    def _format_label(self, label):
        """Formata label para exibição"""
        if label.startswith('quality_'):
            score = label.split('_')[1]
            stars = '⭐' * int(score)
            return f"Qualidade {score} {stars}"
        elif label.startswith('reject_'):
            reason = label.split('_')[1]
            reason_map = {
                'blur': '💫 Desfocada',
                'dark': '🌑 Muito escura',
                'light': '☀️ Muito clara', 
                'cropped': '✂️ Cortada',
                'other': '❓ Outros'
            }
            return reason_map.get(reason, f"Rejeitada ({reason})")
        return label

def main():
    """Função principal para teste interativo"""
    tester = AIPredictionTester()
    
    print("🧠 TESTADOR DE PREDIÇÕES DE IA")
    print("=" * 40)
    
    while True:
        print("\\nOpções:")
        print("1. Treinar novo modelo")
        print("2. Testar em imagens não rotuladas")
        print("3. Validar com dados conhecidos")
        print("4. Sair")
        
        choice = input("\\nEscolha uma opção (1-4): ").strip()
        
        if choice == '1':
            tester.train_current_model()
        elif choice == '2':
            limit = input("Quantas predições mostrar? (padrão: 20): ").strip()
            limit = int(limit) if limit.isdigit() else 20
            tester.predict_unlabeled_samples(limit)
        elif choice == '3':
            tester.test_on_labeled_data()
        elif choice == '4':
            print("👋 Até logo!")
            break
        else:
            print("❌ Opção inválida")

if __name__ == "__main__":
    main()
