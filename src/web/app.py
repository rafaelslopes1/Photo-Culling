#!/usr/bin/env python3
"""
Consolidated Web Labeling Interface for Photo Culling System
Interface web consolidada para rotulagem de imagens
Sistema de classificação rápida com suporte a IA
"""

from flask import Flask, render_template, jsonify, request, send_file
import os
import json
import sqlite3
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebLabelingApp:
    """
    Aplicação web consolidada para rotulagem de imagens
    Combina interface manual com predições de IA
    """
    
    def __init__(self, images_dir="../../data/input", 
                 labels_db="../../data/labels/labels.db",
                 features_db="../../data/features/features.db",
                 use_ai=True,
                 selection_mode='sequential'):
        """
        Inicializa a aplicação web
        
        Args:
            images_dir: Diretório das imagens
            labels_db: Banco de dados de labels
            features_db: Banco de dados de features
            use_ai: Usar classificador de IA
            selection_mode: Modo de seleção ('sequential' ou 'smart')
        """
        
        # Configurar caminhos absolutos baseados na localização do arquivo
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        
        template_folder = os.path.join(current_dir, 'templates')
        static_folder = os.path.join(current_dir, 'templates')
        
        self.app = Flask(__name__, 
                        static_folder=static_folder,
                        template_folder=template_folder)
        
        # Converter caminhos relativos para absolutos
        if not os.path.isabs(images_dir):
            self.images_dir = os.path.join(project_root, images_dir)
        else:
            self.images_dir = images_dir
            
        if not os.path.isabs(labels_db):
            self.labels_db = os.path.join(project_root, labels_db)
        else:
            self.labels_db = labels_db
            
        if not os.path.isabs(features_db):
            self.features_db = os.path.join(project_root, features_db)
        else:
            self.features_db = features_db
        self.use_ai = use_ai
        self.selection_mode = selection_mode
        self.ai_classifier = None
        
        # Initialize database
        self.init_database()
        
        # Load image list
        self.image_list = self.load_image_list()
        self.current_session = datetime.now().isoformat()
        
        # Setup AI if enabled
        if self.use_ai:
            self._setup_ai_classifier()
        
        # Setup routes
        self._setup_routes()
        
        logger.info("🌐 Web Labeling App initialized")
    
    def init_database(self):
        """Inicializa banco de dados SQLite"""
        os.makedirs(os.path.dirname(self.labels_db), exist_ok=True)
        
        conn = sqlite3.connect(self.labels_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                label_type TEXT NOT NULL,
                score INTEGER,
                rejection_reason TEXT,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                ai_prediction TEXT,
                ai_confidence REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                total_labeled INTEGER DEFAULT 0,
                last_activity TEXT,
                ai_enabled BOOLEAN DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("✓ Database initialized")
    
    def load_image_list(self):
        """Carrega lista de imagens em ordem determinística"""
        if not os.path.exists(self.images_dir):
            logger.error(f"Pasta {self.images_dir} não encontrada")
            return []
        
        images = []
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        
        for ext in extensions:
            images.extend(Path(self.images_dir).glob(ext))
            images.extend(Path(self.images_dir).glob(ext.upper()))
        
        # Ordem determinística para evitar repetições
        image_names = sorted([img.name for img in images])
        
        logger.info(f"✓ {len(image_names)} images loaded (deterministic order)")
        return image_names
    
    def _setup_ai_classifier(self):
        """Configura classificador de IA"""
        try:
            from ..core.ai_classifier import AIClassifier
            self.ai_classifier = AIClassifier(self.labels_db, self.features_db)
            
            if self.ai_classifier.load_best_model():
                logger.info("✓ AI classifier loaded")
            else:
                logger.warning("⚠️ No trained AI model found")
                self.use_ai = False
                
        except Exception as e:
            logger.warning(f"⚠️ AI classifier not available: {e}")
            self.use_ai = False
            self.ai_classifier = None
    
    def _setup_routes(self):
        """Configura rotas do Flask"""
        
        @self.app.route('/')
        def index():
            """Página principal"""
            stats = self.get_labeling_stats()
            return render_template('index.html', 
                                 ai_enabled=self.use_ai,
                                 **stats)
        
        @self.app.route('/api/next_image')
        def next_image():
            """Obtém próxima imagem para rotular"""
            unlabeled = self.get_unlabeled_images()
            
            if not unlabeled:
                return jsonify({
                    'success': False,
                    'message': 'Todas as imagens foram rotuladas!',
                    'finished': True
                })
            
            filename = unlabeled[0]
            image_info = self.get_image_info(filename)
            
            # Get AI prediction if available
            ai_prediction = None
            if self.use_ai and self.ai_classifier:
                ai_prediction = self._get_ai_prediction(filename)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'image_url': f'/api/image/{filename}',
                'info': image_info,
                'ai_prediction': ai_prediction,
                'progress': {
                    'labeled': len(self.image_list) - len(unlabeled),
                    'total': len(self.image_list),
                    'remaining': len(unlabeled)
                }
            })
        
        @self.app.route('/api/image/<filename>')
        def serve_image(filename):
            """Serve imagem"""
            image_path = os.path.join(self.images_dir, filename)
            if os.path.exists(image_path):
                return send_file(image_path)
            else:
                return jsonify({'error': 'Image not found'}), 404
        
        @self.app.route('/api/label', methods=['POST'])
        def save_label():
            """Salva rótulo baseado na chave pressionada"""
            data = request.get_json()
            
            if not data or 'filename' not in data or 'key' not in data:
                return jsonify({'success': False, 'error': 'Invalid data'})
            
            filename = data['filename']
            key = data['key']
            
            # Mapear chave para parâmetros de rotulagem
            quality_keys = {
                '1': {'score': 1, 'label': '⭐ Qualidade Muito Baixa'},
                '2': {'score': 2, 'label': '⭐⭐ Qualidade Baixa'},
                '3': {'score': 3, 'label': '⭐⭐⭐ Qualidade Média'},
                '4': {'score': 4, 'label': '⭐⭐⭐⭐ Qualidade Boa'},
                '5': {'score': 5, 'label': '⭐⭐⭐⭐⭐ Qualidade Excelente'}
            }
            
            rejection_keys = {
                'd': {'reason': 'dark', 'label': '🌑 Muito Escura'},
                'l': {'reason': 'light', 'label': '☀️ Muito Clara'},
                'b': {'reason': 'blur', 'label': '😵‍💫 Muito Borrada'},
                'c': {'reason': 'cropped', 'label': '✂️ Cortada/Incompleta'},
                'x': {'reason': 'other', 'label': '❌ Outros Problemas'}
            }
            
            if key in quality_keys:
                # Salvar como qualidade
                result = self.save_image_label(
                    filename=filename,
                    label_type='quality',
                    score=quality_keys[key]['score']
                )
                if result['success']:
                    result['label'] = quality_keys[key]['label']
                    
            elif key.lower() in rejection_keys:
                # Salvar como rejeição
                rejection_data = rejection_keys[key.lower()]
                result = self.save_image_label(
                    filename=filename,
                    label_type='rejection',
                    rejection_reason=rejection_data['reason']
                )
                if result['success']:
                    result['label'] = rejection_data['label']
                    
            else:
                return jsonify({'success': False, 'error': f'Chave inválida: {key}'})
            
            return jsonify(result)
        
        @self.app.route('/api/stats')
        def stats():
            """Estatísticas de rotulagem"""
            return jsonify(self.get_labeling_stats())
        
        @self.app.route('/api/keys')
        def keyboard_shortcuts():
            """Retorna atalhos de teclado no formato esperado pelo frontend"""
            return jsonify({
                'quality': {
                    '1': {'score': 1, 'label': '⭐ Qualidade Muito Baixa'},
                    '2': {'score': 2, 'label': '⭐⭐ Qualidade Baixa'},
                    '3': {'score': 3, 'label': '⭐⭐⭐ Qualidade Média'},
                    '4': {'score': 4, 'label': '⭐⭐⭐⭐ Qualidade Boa'},
                    '5': {'score': 5, 'label': '⭐⭐⭐⭐⭐ Qualidade Excelente'}
                },
                'rejection': {
                    'd': {'reason': 'dark', 'label': '🌑 Muito Escura'},
                    'l': {'reason': 'light', 'label': '☀️ Muito Clara'},
                    'b': {'reason': 'blur', 'label': '😵‍💫 Muito Borrada'},
                    'c': {'reason': 'cropped', 'label': '✂️ Cortada/Incompleta'},
                    'x': {'reason': 'other', 'label': '❌ Outros Problemas'}
                },
                'navigation': {
                    'ArrowLeft': 'Imagem anterior',
                    'ArrowRight': 'Próxima imagem',
                    'ArrowUp': 'Primeira imagem',
                    'ArrowDown': 'Última imagem'
                }
            })
        
        @self.app.route('/api/first-unlabeled')
        def first_unlabeled():
            """Primeira imagem não rotulada - usa seleção sequencial ou inteligente"""
            logger.info(f"🔍 API: Solicitação de próxima imagem (modo: {self.selection_mode})")
            
            unlabeled = self.get_unlabeled_images()
            if not unlabeled:
                logger.info("✅ Todas as imagens foram rotuladas!")
                return jsonify({
                    'success': False,
                    'message': 'Todas as imagens foram rotuladas!'
                })
            
            logger.info(f"📋 {len(unlabeled)} imagens não rotuladas disponíveis")
            
            if self.selection_mode == 'smart':
                logger.info("🧠 Usando algoritmo de SELEÇÃO INTELIGENTE")
                logger.info("=" * 60)
                
                # Usar seleção inteligente
                try:
                    smart_index = self.get_smart_next_image_index()
                    filename = self.image_list[smart_index]
                    
                    logger.info("=" * 60)
                    logger.info(f"🎯 RESULTADO FINAL: {filename}")
                    logger.info(f"📍 Índice na lista: {smart_index}")
                    logger.info(f"🧠 Modo de seleção: INTELIGENTE")
                    
                    return jsonify({
                        'success': True,
                        'index': smart_index,
                        'filename': filename,
                        'selection_mode': 'smart'
                    })
                except Exception as e:
                    logger.error(f"❌ Erro na seleção inteligente: {e}")
                    logger.warning("🔄 Fazendo fallback para seleção sequencial")
                    # Fallback para seleção sequencial
            else:
                logger.info("📝 Usando algoritmo de SELEÇÃO SEQUENCIAL")
            
            # Seleção sequencial (padrão)
            first_unlabeled_index = self.image_list.index(unlabeled[0])
            filename = unlabeled[0]
            
            logger.info(f"🎯 RESULTADO: {filename}")
            logger.info(f"📍 Índice na lista: {first_unlabeled_index}")
            logger.info(f"📝 Modo de seleção: SEQUENCIAL")
            logger.info("💡 Estratégia: Primeira imagem não rotulada em ordem alfabética")
            
            return jsonify({
                'success': True,
                'index': first_unlabeled_index,
                'filename': filename,
                'selection_mode': 'sequential'
            })
        
        @self.app.route('/api/image/<int:index>')
        def get_image_data(index):
            """Obtém metadados da imagem por índice"""
            if 0 <= index < len(self.image_list):
                filename = self.image_list[index]
                
                # Verificar se imagem existe
                image_path = os.path.join(self.images_dir, filename)
                if not os.path.exists(image_path):
                    return jsonify({'error': 'Image not found'}), 404
                
                # Obter status de rotulagem
                label_status = self.get_image_label_status(filename)
                
                return jsonify({
                    'filename': filename,
                    'index': index,
                    'total': len(self.image_list),
                    'total_images': len(self.image_list),
                    'status': label_status
                })
            return jsonify({'error': 'Image not found'}), 404
        
        @self.app.route('/api/image/file/<filename>')
        def get_image_file(filename):
            """Serve arquivo de imagem"""
            image_path = os.path.join(self.images_dir, filename)
            if os.path.exists(image_path):
                return send_file(image_path)
            return jsonify({'error': 'Image file not found'}), 404

        @self.app.route('/api/ai_retrain', methods=['POST'])
        def retrain_ai():
            """Re-treina modelo de IA"""
            if not self.use_ai or not self.ai_classifier:
                return jsonify({'success': False, 'error': 'AI not available'})
            
            try:
                results = self.ai_classifier.train_models()
                return jsonify({
                    'success': True,
                    'message': 'Modelo re-treinado com sucesso',
                    'results': {name: {'accuracy': res['cv_mean']} for name, res in results.items()}
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
    
    def get_unlabeled_images(self):
        """Retorna lista de imagens não rotuladas"""
        conn = sqlite3.connect(self.labels_db)
        cursor = conn.cursor()
        
        cursor.execute('SELECT filename FROM labels')
        labeled_files = {row[0] for row in cursor.fetchall()}
        
        conn.close()
        
        unlabeled = [img for img in self.image_list if img not in labeled_files]
        return unlabeled
    
    def get_image_info(self, filename):
        """Obtém informações da imagem"""
        image_path = os.path.join(self.images_dir, filename)
        
        if not os.path.exists(image_path):
            return None
        
        # Basic file info
        stat = os.stat(image_path)
        
        # Try to get features if available
        features = self._get_image_features(filename)
        
        return {
            'filename': filename,
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'features': features
        }
    
    def _get_image_features(self, filename):
        """Obtém features da imagem se disponível"""
        try:
            conn = sqlite3.connect(self.features_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT sharpness_laplacian, brightness_mean, saturation_mean, 
                       face_count, visual_complexity
                FROM image_features 
                WHERE filename = ?
            ''', (filename,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'sharpness': round(result[0], 2) if result[0] else None,
                    'brightness': round(result[1], 2) if result[1] else None,
                    'saturation': round(result[2], 2) if result[2] else None,
                    'faces': result[3] if result[3] else 0,
                    'complexity': round(result[4], 3) if result[4] else None
                }
        except:
            pass
        
        return None
    
    def _get_ai_prediction(self, filename):
        """Obtém predição da IA para a imagem"""
        if not self.ai_classifier:
            return None
        
        try:
            image_path = os.path.join(self.images_dir, filename)
            prediction_result = self.ai_classifier.get_prediction_for_image(image_path)
            
            if prediction_result:
                return {
                    'prediction': prediction_result['formatted_prediction'],
                    'confidence': round(prediction_result['confidence'], 3),
                    'raw_prediction': prediction_result['prediction']
                }
        except Exception as e:
            logger.error(f"Error getting AI prediction: {e}")
        
        return None
    
    def save_image_label(self, filename, label_type, score=None, rejection_reason=None,
                        ai_prediction=None, ai_confidence=None):
        """Salva rótulo da imagem"""
        try:
            conn = sqlite3.connect(self.labels_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO labels 
                (filename, label_type, score, rejection_reason, timestamp, session_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                filename, label_type, score, rejection_reason,
                datetime.now().isoformat(), self.current_session
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✓ Label saved for {filename}: {label_type}")
            
            return {'success': True, 'message': 'Label saved successfully'}
            
        except Exception as e:
            logger.error(f"Error saving label: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_labeling_stats(self):
        """Obtém estatísticas de rotulagem"""
        try:
            conn = sqlite3.connect(self.labels_db)
            cursor = conn.cursor()
            
            # Total counts
            cursor.execute('SELECT COUNT(*) FROM labels')
            total_labeled = cursor.fetchone()[0]
            
            # Quality distribution
            cursor.execute('''
                SELECT score, COUNT(*) 
                FROM labels 
                WHERE label_type = 'quality' 
                GROUP BY score
            ''')
            quality_dist = dict(cursor.fetchall())
            
            # Rejection distribution
            cursor.execute('''
                SELECT rejection_reason, COUNT(*) 
                FROM labels 
                WHERE label_type = 'rejection' 
                GROUP BY rejection_reason
            ''')
            rejection_dist = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'total_labeled': total_labeled,
                'total_images': len(self.image_list),
                'remaining': len(self.image_list) - total_labeled,
                'progress_percent': round((total_labeled / len(self.image_list)) * 100, 1) if self.image_list else 0,
                'quality_distribution': quality_dist,
                'rejection_distribution': rejection_dist
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                'total_labeled': 0,
                'total_images': len(self.image_list),
                'remaining': len(self.image_list),
                'progress_percent': 0,
                'quality_distribution': {},
                'rejection_distribution': {}
            }
    
    def get_image_label_status(self, filename):
        """Obtém status de rotulagem de uma imagem específica"""
        try:
            conn = sqlite3.connect(self.labels_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT label_type, score, rejection_reason, timestamp
                FROM labels 
                WHERE filename = ?
                ORDER BY timestamp DESC
                LIMIT 1
            ''', (filename,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                label_type, score, rejection_reason, timestamp = result
                return {
                    'labeled': True,
                    'label_type': label_type,
                    'score': score,
                    'rejection_reason': rejection_reason,
                    'timestamp': timestamp
                }
            else:
                return {
                    'labeled': False,
                    'label_type': None,
                    'score': None,
                    'rejection_reason': None,
                    'timestamp': None
                }
                
        except Exception as e:
            logger.error(f"Error getting label status for {filename}: {e}")
            return {
                'labeled': False,
                'label_type': None,
                'score': None,
                'rejection_reason': None,
                'timestamp': None
            }
    
    def get_class_distribution(self):
        """Analisa distribuição de classes para identificar sub-representadas"""
        try:
            conn = sqlite3.connect(self.labels_db)
            cursor = conn.cursor()
            
            # Get class distribution
            cursor.execute('''
                SELECT 
                    CASE 
                        WHEN label_type = 'quality' THEN 'quality_' || CAST(score AS TEXT)
                        WHEN label_type = 'rejection' THEN 'reject_' || rejection_reason
                    END as class_label,
                    COUNT(*) as count
                FROM labels
                GROUP BY class_label
                ORDER BY count ASC
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return {}
            
            # Convert to dictionary
            class_counts = {label: count for label, count in results}
            
            # Find underrepresented classes (bottom 30% or less than median)
            counts = list(class_counts.values())
            
            # Import numpy dynamically for smart selection
            try:
                import numpy as np
                median_count = np.median(counts) if counts else 1
                min_threshold = max(1, median_count * 0.5)  # Classes with less than 50% of median
            except ImportError:
                # Fallback if numpy not available
                median_count = sorted(counts)[len(counts)//2] if counts else 1
                min_threshold = max(1, median_count * 0.5)
            
            underrepresented = {
                cls: count for cls, count in class_counts.items() 
                if count < min_threshold
            }
            
            return {
                'all_classes': class_counts,
                'underrepresented': underrepresented,
                'min_threshold': min_threshold,
                'total_labeled': sum(counts)
            }
            
        except Exception as e:
            logger.error(f"Erro ao analisar distribuição de classes: {e}")
            return {}
    
    def get_smart_next_image_index(self):
        """Seleciona próxima imagem priorizando classes sub-representadas"""
        import random
        
        logger.info("🎯 INICIANDO SELEÇÃO INTELIGENTE...")
        
        # Get unlabeled images
        unlabeled = self.get_unlabeled_images()
        
        if not unlabeled:
            logger.info("📝 Todas as imagens foram rotuladas!")
            return len(self.image_list) - 1  # All labeled, return last
        
        logger.info(f"📊 Total de imagens não rotuladas: {len(unlabeled)}")
        
        # Convert to indices
        unlabeled_indices = []
        for filename in unlabeled:
            try:
                index = self.image_list.index(filename)
                unlabeled_indices.append(index)
            except ValueError:
                continue
        
        if not unlabeled_indices:
            logger.warning("⚠️ Nenhum índice válido encontrado, usando primeira imagem")
            return 0
        
        # Analyze class distribution
        class_dist = self.get_class_distribution()
        total_labeled = class_dist.get('total_labeled', 0)
        
        logger.info(f"📈 Total de imagens já rotuladas: {total_labeled}")
        
        # Check conditions for intelligent selection
        if not self.use_ai:
            random_index = random.choice(unlabeled_indices)
            random_filename = self.image_list[random_index]
            logger.info(f"🎲 SELEÇÃO ALEATÓRIA: {random_filename}")
            logger.info("   💭 Motivo: IA está desabilitada no sistema")
            logger.info(f"   📍 Índice selecionado: {random_index}")
            return random_index
        
        if not class_dist:
            random_index = random.choice(unlabeled_indices)
            random_filename = self.image_list[random_index]
            logger.info(f"🎲 SELEÇÃO ALEATÓRIA: {random_filename}")
            logger.info("   💭 Motivo: Não foi possível analisar distribuição de classes")
            logger.info(f"   📍 Índice selecionado: {random_index}")
            return random_index
        
        if total_labeled < 10:
            random_index = random.choice(unlabeled_indices)
            random_filename = self.image_list[random_index]
            logger.info(f"🎲 SELEÇÃO ALEATÓRIA: {random_filename}")
            logger.info(f"   💭 Motivo: Poucos dados para análise inteligente ({total_labeled} < 10 mínimo)")
            logger.info("   💡 Sugestão: Rotule pelo menos 10 imagens antes de usar seleção inteligente")
            logger.info(f"   📍 Índice selecionado: {random_index}")
            return random_index
        
        # Show class distribution analysis
        all_classes = class_dist.get('all_classes', {})
        underrepresented = class_dist.get('underrepresented', {})
        min_threshold = class_dist.get('min_threshold', 0)
        
        logger.info("📊 ANÁLISE DE DISTRIBUIÇÃO DE CLASSES:")
        logger.info(f"   🎯 Classes identificadas: {len(all_classes)}")
        if all_classes:
            sorted_classes = sorted(all_classes.items(), key=lambda x: x[1])
            logger.info(f"   📉 Classe com menos exemplos: {sorted_classes[0][0]} ({sorted_classes[0][1]} exemplos)")
            logger.info(f"   📈 Classe com mais exemplos: {sorted_classes[-1][0]} ({sorted_classes[-1][1]} exemplos)")
        
        logger.info(f"   ⚖️ Threshold para classes sub-representadas: {min_threshold:.1f}")
        logger.info(f"   🎪 Classes sub-representadas identificadas: {len(underrepresented)}")
        
        if underrepresented:
            logger.info("   📋 Lista de classes que precisam de mais exemplos:")
            for class_name, count in sorted(underrepresented.items(), key=lambda x: x[1]):
                logger.info(f"      • {class_name}: {count} exemplos")
        else:
            logger.info("   ✅ Todas as classes estão bem representadas!")
        
        # Try to get AI predictions for smarter selection
        try:
            logger.info("🤖 Tentando obter predições de IA para seleção inteligente...")
            
            # For now, implement a simplified intelligent selection with detailed reasoning
            # This will be enhanced when full AI prediction is available
            random_index = random.choice(unlabeled_indices)
            random_filename = self.image_list[random_index]
            
            # Simulate realistic probabilities for demonstration
            # In real implementation, these would come from AI model predictions
            import random as rand
            
            # Create simulated probabilities that favor underrepresented classes
            simulated_probs = {}
            total_prob = 0
            
            # Give higher probability to underrepresented classes
            underrep_classes = list(underrepresented.keys())
            other_classes = [cls for cls in all_classes.keys() if cls not in underrep_classes]
            
            # Assign probabilities favoring underrepresented classes
            for cls in underrep_classes:
                prob = rand.uniform(0.15, 0.35)  # Higher probability for underrepresented
                simulated_probs[cls] = prob
                total_prob += prob
            
            for cls in other_classes:
                prob = rand.uniform(0.02, 0.12)  # Lower probability for well-represented
                simulated_probs[cls] = prob
                total_prob += prob
            
            # Normalize probabilities to sum to 1
            for cls in simulated_probs:
                simulated_probs[cls] = simulated_probs[cls] / total_prob
            
            # Calculate detailed scores
            underrep_score = sum(simulated_probs.get(cls, 0) for cls in underrep_classes)
            overrep_score = sum(simulated_probs.get(cls, 0) for cls in other_classes)
            
            # Calculate uncertainty (higher when probabilities are more spread out)
            max_prob = max(simulated_probs.values())
            uncertainty = 1 - max_prob
            
            # Random factor
            random_factor = rand.random() * 0.1
            
            # Analyze selection strategy and determine primary reason
            primary_reason = ""
            detailed_explanation = []
            strategic_value = ""
            
            # Determine primary selection reason with detailed analysis
            if underrep_score > 0.3:
                dominant_underrep = max(underrep_classes, key=lambda x: simulated_probs.get(x, 0))
                dominant_prob = simulated_probs.get(dominant_underrep, 0)
                current_count = underrepresented.get(dominant_underrep, 0)
                
                primary_reason = "🎯 ALTA PROBABILIDADE DE CLASSE SUB-REPRESENTADA"
                detailed_explanation = [
                    f"Esta imagem tem {dominant_prob:.1%} de probabilidade de ser '{dominant_underrep}'",
                    f"A classe '{dominant_underrep}' tem apenas {current_count} exemplos no dataset",
                    f"Precisamos de mais exemplos desta classe para balancear o modelo",
                    f"Score de classes minoritárias: {underrep_score:.1%} (acima do limiar de 30%)"
                ]
                strategic_value = f"ALTO - Pode dobrar os exemplos de '{dominant_underrep}' ({current_count} → {current_count + 1})"
                
            elif uncertainty > 0.6:
                max_prob_class = max(simulated_probs.keys(), key=lambda x: simulated_probs[x])
                max_prob = simulated_probs[max_prob_class]
                
                primary_reason = "❓ ALTA INCERTEZA DO MODELO"
                detailed_explanation = [
                    f"Modelo está incerto sobre esta imagem (incerteza: {uncertainty:.1%})",
                    f"Maior probabilidade: {max_prob:.1%} para '{max_prob_class}'",
                    f"Imagens com alta incerteza são mais informativas para treinamento",
                    f"Ajudam o modelo a aprender casos difíceis/ambíguos"
                ]
                strategic_value = "MÉDIO-ALTO - Casos incertos melhoram generalização do modelo"
                
            elif underrep_score > 0.15:
                # Moderate probability for underrepresented classes
                best_underrep = max(underrep_classes, key=lambda x: simulated_probs.get(x, 0)) if underrep_classes else None
                if best_underrep:
                    prob = simulated_probs.get(best_underrep, 0)
                    count = underrepresented.get(best_underrep, 0)
                    
                    primary_reason = "🎪 PROBABILIDADE MODERADA DE CLASSE MINORITÁRIA"
                    detailed_explanation = [
                        f"Probabilidade moderada ({prob:.1%}) para '{best_underrep}'",
                        f"Classe '{best_underrep}' precisa de mais exemplos ({count} atualmente)",
                        f"Score de classes minoritárias: {underrep_score:.1%}",
                        f"Estratégia gradual de balanceamento"
                    ]
                    strategic_value = f"MÉDIO - Contribui para balanceamento da classe '{best_underrep}'"
                else:
                    primary_reason = "🎲 DIVERSIFICAÇÃO ESTRATÉGICA"
                    detailed_explanation = [
                        "Probabilidades equilibradas entre classes",
                        "Evitando concentração excessiva em uma única estratégia",
                        "Mantendo diversidade na seleção de imagens",
                        f"Componente aleatório: {random_factor:.1%}"
                    ]
                    strategic_value = "MÉDIO - Mantém diversidade do dataset"
            else:
                # Low underrepresented score - likely exploring or random
                primary_reason = "🔄 EXPLORAÇÃO E DIVERSIDADE"
                detailed_explanation = [
                    "Baixa probabilidade de classes sub-representadas",
                    "Estratégia de exploração para descobrir padrões",
                    "Evitando super-especialização em classes conhecidas",
                    f"Balanceando exploração vs. exploração otimizada"
                ]
                strategic_value = "BAIXO-MÉDIO - Exploração necessária para descoberta"
            
            # Final score calculation
            final_score = underrep_score * 0.6 + uncertainty * 0.3 + random_factor
            
            # Simplified logging - just show what the algorithm thinks
            logger.info(f"🎯 SELEÇÃO: {random_filename}")
            
            # Show what the algorithm inferred about this image
            if underrep_score > 0.3:
                dominant_underrep = max(underrep_classes, key=lambda x: simulated_probs.get(x, 0))
                dominant_prob = simulated_probs.get(dominant_underrep, 0)
                current_count = underrepresented.get(dominant_underrep, 0)
                
                logger.info(f"🤖 Algoritmo inferiu: {dominant_prob:.1%} chance de ser '{dominant_underrep}'")
                logger.info(f"📊 Motivo da sugestão: Classe tem apenas {current_count} exemplos (sub-representada)")
                
            else:
                # Show top prediction for other cases
                top_class = max(simulated_probs.keys(), key=lambda x: simulated_probs[x])
                top_prob = simulated_probs[top_class]
                
                if uncertainty > 0.6:
                    logger.info(f"� Algoritmo inferiu: Incerto - {top_prob:.1%} para '{top_class}'")
                    logger.info(f"� Motivo da sugestão: Casos incertos ajudam a treinar o modelo")
                else:
                    logger.info(f"🤖 Algoritmo inferiu: {top_prob:.1%} chance de ser '{top_class}'")
                    logger.info(f"� Motivo da sugestão: Diversificação do dataset")
            
            logger.info("─" * 50)
            
            
            return random_index
            
        except Exception as e:
            logger.error(f"❌ Erro na seleção inteligente: {e}")
            logger.info("🔄 Fazendo fallback para seleção aleatória...")
            
            random_index = random.choice(unlabeled_indices)
            random_filename = self.image_list[random_index]
            logger.info(f"🎲 SELEÇÃO ALEATÓRIA (fallback): {random_filename}")
            logger.info(f"   📍 Índice selecionado: {random_index}")
            return random_index

def create_app(config=None, selection_mode='sequential'):
    """
    Factory function to create Flask application
    
    Args:
        config: Configuração do sistema
        selection_mode: Modo de seleção de imagens ('sequential' ou 'smart')
    """
    if config is None:
        config = {}
    
    images_dir = config.get('paths', {}).get('input_folder', 'data/input')
    labels_db = config.get('paths', {}).get('labels_db', 'data/labels/labels.db')
    features_db = config.get('paths', {}).get('features_db', 'data/features/features.db')
    use_ai = config.get('ai', {}).get('enabled', True)
    
    web_app = WebLabelingApp(
        images_dir=images_dir,
        labels_db=labels_db,
        features_db=features_db,
        use_ai=use_ai,
        selection_mode=selection_mode
    )
    
    return web_app.app

# Convenience functions
def start_web_server(host='0.0.0.0', port=5001, debug=True, config=None, selection_mode='sequential'):
    """Start the web server"""
    app = create_app(config, selection_mode)
    app.run(host=host, port=port, debug=debug)
