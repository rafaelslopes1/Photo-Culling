#!/usr/bin/env python3
"""
Backend Flask para Rotulagem de Imagens
Sistema de classificação rápida com teclado
"""

from flask import Flask, render_template, jsonify, request, send_file
import os
import json
import random
from datetime import datetime
import sqlite3
from pathlib import Path

app = Flask(__name__)

# Configurações
IMAGES_DIR = "../input"  # Pasta com as imagens
DB_PATH = "data/labels.db"
LABELS_JSON = "data/labels.json"

# Mapeamento de teclas para rótulos
QUALITY_KEYS = {
    '1': {'score': 1, 'label': '⭐ Qualidade Muito Baixa'},
    '2': {'score': 2, 'label': '⭐⭐ Qualidade Baixa'},
    '3': {'score': 3, 'label': '⭐⭐⭐ Qualidade Média'},
    '4': {'score': 4, 'label': '⭐⭐⭐⭐ Qualidade Boa'},
    '5': {'score': 5, 'label': '⭐⭐⭐⭐⭐ Qualidade Excelente'}
}

REJECTION_KEYS = {
    'd': {'reason': 'dark', 'label': '🌑 Muito Escura'},
    'l': {'reason': 'light', 'label': '☀️ Muito Clara'},
    'b': {'reason': 'blur', 'label': '😵‍💫 Muito Borrada'},
    'c': {'reason': 'cropped', 'label': '✂️ Cortada/Incompleta'},
    'x': {'reason': 'other', 'label': '❌ Outros Problemas'}
}

class ImageLabeler:
    def __init__(self):
        self.init_database()
        self.image_list = self.load_image_list()
        self.current_session = datetime.now().isoformat()
    
    def init_database(self):
        """Inicializa banco de dados SQLite"""
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                label_type TEXT NOT NULL,  -- 'quality' ou 'rejection'
                score INTEGER,             -- 1-5 para qualidade, NULL para rejeição
                rejection_reason TEXT,     -- motivo da rejeição
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time TEXT NOT NULL,
                total_labeled INTEGER DEFAULT 0,
                last_activity TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✓ Banco de dados inicializado")
    
    def load_image_list(self):
        """Carrega lista de imagens"""
        if not os.path.exists(IMAGES_DIR):
            print(f"❌ Pasta {IMAGES_DIR} não encontrada")
            return []
        
        images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            images.extend(Path(IMAGES_DIR).glob(ext))
        
        # Converte para lista de nomes e embaralha
        image_names = [img.name for img in images]
        random.shuffle(image_names)
        
        print(f"✓ {len(image_names)} imagens carregadas")
        return image_names
    
    def get_image_status(self, filename):
        """Verifica se imagem já foi rotulada"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT label_type, score, rejection_reason, timestamp 
            FROM labels WHERE filename = ?
        ''', (filename,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            label_type, score, rejection_reason, timestamp = result
            if label_type == 'quality':
                return {
                    'labeled': True,
                    'type': 'quality',
                    'score': score,
                    'label': QUALITY_KEYS[str(score)]['label'],
                    'timestamp': timestamp
                }
            else:
                return {
                    'labeled': True,
                    'type': 'rejection',
                    'reason': rejection_reason,
                    'label': next(v['label'] for k, v in REJECTION_KEYS.items() if v['reason'] == rejection_reason),
                    'timestamp': timestamp
                }
        
        return {'labeled': False}
    
    def save_label(self, filename, key):
        """Salva rótulo no banco de dados"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        # Remove rótulo anterior se existir
        cursor.execute('DELETE FROM labels WHERE filename = ?', (filename,))
        
        if key in QUALITY_KEYS:
            # Rótulo de qualidade
            score = QUALITY_KEYS[key]['score']
            cursor.execute('''
                INSERT INTO labels (filename, label_type, score, timestamp, session_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (filename, 'quality', score, timestamp, self.current_session))
            
            label = QUALITY_KEYS[key]['label']
            
        elif key in REJECTION_KEYS:
            # Rótulo de rejeição
            reason = REJECTION_KEYS[key]['reason']
            cursor.execute('''
                INSERT INTO labels (filename, label_type, rejection_reason, timestamp, session_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (filename, 'rejection', reason, timestamp, self.current_session))
            
            label = REJECTION_KEYS[key]['label']
        
        else:
            conn.close()
            return None
        
        conn.commit()
        conn.close()
        
        # Atualiza também arquivo JSON para backup
        self.update_json_backup()
        
        return {
            'success': True,
            'label': label,
            'timestamp': timestamp
        }
    
    def update_json_backup(self):
        """Atualiza backup em JSON"""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT filename, label_type, score, rejection_reason, timestamp
                FROM labels ORDER BY timestamp
            ''')
            
            labels = {}
            for row in cursor.fetchall():
                filename, label_type, score, rejection_reason, timestamp = row
                labels[filename] = {
                    'type': label_type,
                    'score': score,
                    'rejection_reason': rejection_reason,
                    'timestamp': timestamp
                }
            
            conn.close()
            
            os.makedirs(os.path.dirname(LABELS_JSON), exist_ok=True)
            with open(LABELS_JSON, 'w') as f:
                json.dump(labels, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Erro ao atualizar backup JSON: {e}")
    
    def get_stats(self):
        """Estatísticas de rotulagem"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Total de imagens
        total_images = len(self.image_list)
        
        # Total rotuladas
        cursor.execute('SELECT COUNT(*) FROM labels')
        total_labeled = cursor.fetchone()[0]
        
        # Por qualidade
        cursor.execute('SELECT score, COUNT(*) FROM labels WHERE label_type = "quality" GROUP BY score')
        quality_stats = dict(cursor.fetchall())
        
        # Por rejeição
        cursor.execute('SELECT rejection_reason, COUNT(*) FROM labels WHERE label_type = "rejection" GROUP BY rejection_reason')
        rejection_stats = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_images': total_images,
            'total_labeled': total_labeled,
            'remaining': total_images - total_labeled,
            'progress_percent': round((total_labeled / total_images) * 100, 1) if total_images > 0 else 0,
            'quality_stats': quality_stats,
            'rejection_stats': rejection_stats
        }

# Instância global
labeler = ImageLabeler()

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/api/image/<int:index>')
def get_image_info(index):
    """Informações sobre uma imagem específica"""
    if not (0 <= index < len(labeler.image_list)):
        return jsonify({'error': 'Índice inválido'}), 400
    
    filename = labeler.image_list[index]
    status = labeler.get_image_status(filename)
    
    return jsonify({
        'index': index,
        'filename': filename,
        'total': len(labeler.image_list),
        'status': status,
        'image_url': f'/api/image/file/{filename}'
    })

@app.route('/api/image/file/<filename>')
def serve_image(filename):
    """Serve arquivo de imagem"""
    image_path = os.path.join(IMAGES_DIR, filename)
    if os.path.exists(image_path):
        return send_file(image_path)
    return "Imagem não encontrada", 404

@app.route('/api/label', methods=['POST'])
def save_label():
    """Salva rótulo de uma imagem"""
    data = request.get_json()
    filename = data.get('filename')
    key = data.get('key')
    
    if not filename or not key:
        return jsonify({'error': 'Dados inválidos'}), 400
    
    result = labeler.save_label(filename, key)
    
    if result:
        return jsonify(result)
    else:
        return jsonify({'error': 'Chave inválida'}), 400

@app.route('/api/stats')
def get_stats():
    """Estatísticas de progresso"""
    return jsonify(labeler.get_stats())

@app.route('/api/first-unlabeled')
def get_first_unlabeled():
    """Retorna o índice da primeira imagem não rotulada"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    for i, filename in enumerate(labeler.image_list):
        cursor.execute('SELECT label_type FROM labels WHERE filename = ?', (filename,))
        result = cursor.fetchone()
        if result is None:  # Não rotulada
            conn.close()
            return jsonify({'index': i})
    
    conn.close()
    # Se todas estão rotuladas, retorna a última
    return jsonify({'index': len(labeler.image_list) - 1})

@app.route('/api/labeled-images')
def get_labeled_images():
    """Retorna lista de imagens rotuladas para revisão"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    labeled_images = []
    for i, filename in enumerate(labeler.image_list):
        cursor.execute('''
            SELECT label_type, score, rejection_reason, timestamp 
            FROM labels WHERE filename = ?
        ''', (filename,))
        result = cursor.fetchone()
        if result:
            label_type, score, rejection_reason, timestamp = result
            
            # Constrói o label baseado no tipo
            if label_type == 'quality':
                label = QUALITY_KEYS[str(score)]['label']
            else:
                label = next(v['label'] for k, v in REJECTION_KEYS.items() if v['reason'] == rejection_reason)
            
            labeled_images.append({
                'index': i,
                'filename': filename,
                'label': label,
                'timestamp': timestamp
            })
    
    conn.close()
    return jsonify({'labeled_images': labeled_images, 'total': len(labeled_images)})

@app.route('/api/keys')
def get_key_mappings():
    """Mapeamento de teclas"""
    return jsonify({
        'quality': QUALITY_KEYS,
        'rejection': REJECTION_KEYS,
        'navigation': {
            'ArrowLeft': 'Imagem anterior',
            'ArrowRight': 'Próxima imagem',
            'ArrowUp': 'Primeira imagem',
            'ArrowDown': 'Última imagem'
        }
    })

if __name__ == '__main__':
    print("🚀 Iniciando servidor de rotulagem de imagens...")
    print(f"📁 Pasta de imagens: {IMAGES_DIR}")
    print(f"🗄️ Banco de dados: {DB_PATH}")
    print("🌐 Acesse: http://localhost:5002")
    print("\n⌨️ Teclas disponíveis:")
    print("   1-5: Qualidade (estrelas)")
    print("   D: Muito escura, L: Muito clara, B: Borrada, C: Cortada, X: Outros")
    print("   ←→: Navegação")
    
    app.run(debug=True, host='0.0.0.0', port=5002)
