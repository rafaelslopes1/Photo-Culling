#!/usr/bin/env python3
"""
Photo Culling Web App v2.0 - Expert Evaluation System (Simplified)
Sistema de coleta de avaliações de especialistas para treinamento de IA
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import json
import sys
import logging
from pathlib import Path

app = Flask(__name__)
app.secret_key = 'photo_culling_expert_system_v2'

# Database configuration
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{basedir}/expert_evaluations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple image manager for web app (without heavy dependencies)
class SimpleImageManager:
    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    def get_available_images(self):
        """Get list of available images for evaluation"""
        images = []
        if self.image_dir.exists():
            for img_file in self.image_dir.iterdir():
                if img_file.suffix.lower() in self.supported_extensions:
                    images.append({
                        'filename': img_file.name,
                        'path': str(img_file),
                        'size': img_file.stat().st_size
                    })
        return sorted(images, key=lambda x: x['filename'])

# Database Models
class ExpertEvaluation(db.Model):
    """Model for storing expert evaluations"""
    id = db.Column(db.Integer, primary_key=True)
    image_filename = db.Column(db.String(255), nullable=False)
    evaluator_id = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Ratings (1-5 scale)
    overall_quality = db.Column(db.Integer)
    global_sharpness = db.Column(db.Integer)
    person_sharpness = db.Column(db.Integer)
    exposure_quality = db.Column(db.Integer)
    composition_quality = db.Column(db.Integer)
    emotional_impact = db.Column(db.Integer)
    technical_execution = db.Column(db.Integer)
    
    # Categorical assessments
    environment_lighting = db.Column(db.String(50))  # muito_escuro, levemente_escuro, ideal, levemente_claro, muito_claro
    person_lighting = db.Column(db.String(50))       # pessoa_muito_escura, pessoa_levemente_escura, ideal, pessoa_levemente_clara, pessoa_estourada
    person_sharpness_level = db.Column(db.String(50)) # muito_nitida, nitida, levemente_desfocada, moderadamente_desfocada, muito_desfocada
    person_position = db.Column(db.String(50))       # centralizada, esquerda, direita, terco_superior, terco_inferior
    eyes_quality = db.Column(db.String(50))          # muito_nitidos, nitidos, levemente_desfocados, desfocados, fechados_nao_visiveis
    
    # Contextual information
    people_count = db.Column(db.String(50))          # sem_pessoas, 1_pessoa, 2_pessoas, 3_5_pessoas, 6_mais_pessoas
    photo_context = db.Column(db.String(50))         # interno, externo, luz_natural, luz_artificial, contraluz, golden_hour
    technical_issues = db.Column(db.Text)            # JSON array of issues
    
    # Binary decisions
    approve_for_portfolio = db.Column(db.Boolean)
    approve_for_client = db.Column(db.Boolean)
    approve_for_social = db.Column(db.Boolean)
    needs_editing = db.Column(db.Boolean)
    complete_reject = db.Column(db.Boolean)
    
    # Additional data
    issues = db.Column(db.Text)  # JSON string of categorical issues
    confidence_level = db.Column(db.Float)
    evaluation_time_seconds = db.Column(db.Integer)
    comments = db.Column(db.Text)
    
    def to_dict(self):
        """Convert evaluation to dictionary"""
        return {
            'id': self.id,
            'image_filename': self.image_filename,
            'evaluator_id': self.evaluator_id,
            'timestamp': self.timestamp.isoformat(),
            'ratings': {
                'overall_quality': self.overall_quality,
                'global_sharpness': self.global_sharpness,
                'person_sharpness': self.person_sharpness,
                'exposure_quality': self.exposure_quality,
                'composition_quality': self.composition_quality,
                'emotional_impact': self.emotional_impact,
                'technical_execution': self.technical_execution
            },
            'categorical_assessments': {
                'environment_lighting': self.environment_lighting,
                'person_lighting': self.person_lighting,
                'person_sharpness_level': self.person_sharpness_level,
                'person_position': self.person_position,
                'eyes_quality': self.eyes_quality,
                'people_count': self.people_count,
                'photo_context': self.photo_context,
                'technical_issues': json.loads(self.technical_issues) if self.technical_issues else []
            },
            'decisions': {
                'approve_for_portfolio': self.approve_for_portfolio,
                'approve_for_client': self.approve_for_client,
                'approve_for_social': self.approve_for_social,
                'needs_editing': self.needs_editing,
                'complete_reject': self.complete_reject
            },
            'issues': json.loads(self.issues) if self.issues else {},
            'confidence_level': self.confidence_level,
            'evaluation_time_seconds': self.evaluation_time_seconds,
            'comments': self.comments
        }

class EvaluationSession(db.Model):
    """Model for tracking evaluation sessions"""
    id = db.Column(db.Integer, primary_key=True)
    evaluator_id = db.Column(db.String(50), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    images_evaluated = db.Column(db.Integer, default=0)
    total_time_seconds = db.Column(db.Integer, default=0)

# Initialize simple image manager
project_root = Path(__file__).parent.parent.parent
image_manager = SimpleImageManager(project_root / 'data' / 'input')

# Routes
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page for evaluators"""
    if request.method == 'POST':
        evaluator_id = request.form.get('evaluator_id')
        if evaluator_id:
            session['evaluator_id'] = evaluator_id
            # Create evaluation session
            eval_session = EvaluationSession(evaluator_id=evaluator_id)
            db.session.add(eval_session)
            db.session.commit()
            session['session_id'] = eval_session.id
            return redirect(url_for('evaluate'))
    
    return render_template('login.html')

@app.route('/evaluate')
def evaluate():
    """Main evaluation interface"""
    if 'evaluator_id' not in session:
        return redirect(url_for('login'))
    
    return render_template('evaluate_v2.html')

@app.route('/api/images')
def get_images():
    """API endpoint to get available images"""
    images = image_manager.get_available_images()
    
    # Filter out already evaluated images (optional)
    evaluator_id = session.get('evaluator_id')
    if evaluator_id:
        evaluated = db.session.query(ExpertEvaluation.image_filename).filter_by(
            evaluator_id=evaluator_id
        ).all()
        evaluated_filenames = {e[0] for e in evaluated}
        images = [img for img in images if img['filename'] not in evaluated_filenames]
    
    return jsonify({
        'success': True,
        'images': images[:50]  # Limit to 50 images per session
    })

@app.route('/api/image/<filename>')
def serve_image(filename):
    """Serve image files"""
    image_path = project_root / 'data' / 'input' / filename
    if image_path.exists():
        return send_file(str(image_path))
    return jsonify({'error': 'Image not found'}), 404

@app.route('/api/evaluate', methods=['POST'])
def submit_evaluation():
    """Submit expert evaluation"""
    if 'evaluator_id' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    
    try:
        data = request.get_json()
        ratings = data.get('ratings', {})
        categorical = data.get('categorical_assessments', {})
        decisions = data.get('decisions', {})
        
        # Create evaluation record
        evaluation = ExpertEvaluation(
            image_filename=data['image_filename'],
            evaluator_id=session['evaluator_id'],
            
            # Ratings
            overall_quality=ratings.get('overall_quality'),
            global_sharpness=ratings.get('global_sharpness'),
            person_sharpness=ratings.get('person_sharpness'),
            exposure_quality=ratings.get('exposure_quality'),
            composition_quality=ratings.get('composition_quality'),
            emotional_impact=ratings.get('emotional_impact'),
            technical_execution=ratings.get('technical_execution'),
            
            # Categorical assessments
            environment_lighting=categorical.get('environment_lighting'),
            person_lighting=categorical.get('person_lighting'),
            person_sharpness_level=categorical.get('person_sharpness_level'),
            person_position=categorical.get('person_position'),
            eyes_quality=categorical.get('eyes_quality'),
            people_count=categorical.get('people_count'),
            photo_context=categorical.get('photo_context'),
            technical_issues=json.dumps(categorical.get('technical_issues', [])),
            
            # Decisions
            approve_for_portfolio=decisions.get('approve_for_portfolio'),
            approve_for_client=decisions.get('approve_for_client'),
            approve_for_social=decisions.get('approve_for_social'),
            needs_editing=decisions.get('needs_editing'),
            complete_reject=decisions.get('complete_reject'),
            
            # Additional data
            issues=json.dumps(data.get('issues', {})),
            confidence_level=data.get('confidence_level'),
            evaluation_time_seconds=data.get('evaluation_time_seconds'),
            comments=data.get('comments', '')
        )
        
        db.session.add(evaluation)
        db.session.commit()
        
        # Update session statistics
        if 'session_id' in session:
            eval_session = EvaluationSession.query.get(session['session_id'])
            if eval_session:
                eval_session.images_evaluated += 1
                eval_session.total_time_seconds += data.get('evaluation_time_seconds', 0)
                db.session.commit()
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Error submitting evaluation: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get evaluation statistics"""
    if 'evaluator_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    evaluator_id = session['evaluator_id']
    
    total_evaluations = ExpertEvaluation.query.filter_by(evaluator_id=evaluator_id).count()
    
    return jsonify({
        'total_evaluations': total_evaluations,
        'evaluator_id': evaluator_id
    })

@app.route('/dashboard')
def dashboard():
    """Dashboard with evaluation statistics"""
    if 'evaluator_id' not in session:
        return redirect(url_for('index'))
    
    try:
        # Basic statistics
        total_avaliacoes = ExpertEvaluation.query.count()
        
        if total_avaliacoes == 0:
            return render_template('dashboard.html', 
                                 total_avaliacoes=0,
                                 media_qualidade=0,
                                 rejeitadas=0,
                                 media_confianca=0,
                                 por_pessoas=[],
                                 por_contexto=[])
        
        # Average quality
        media_qualidade = db.session.query(db.func.avg(ExpertEvaluation.overall_quality)).scalar()
        media_qualidade = round(media_qualidade, 2) if media_qualidade else 0
        
        # Rejected count
        rejeitadas = ExpertEvaluation.query.filter_by(complete_reject=True).count()
        
        # Average confidence
        media_confianca = db.session.query(db.func.avg(ExpertEvaluation.confidence_level)).scalar()
        media_confianca = round(media_confianca, 1) if media_confianca else 0
        
        # Analysis by people count
        por_pessoas = db.session.query(
            ExpertEvaluation.people_count,
            db.func.count().label('quantidade'),
            db.func.round(db.func.avg(ExpertEvaluation.overall_quality), 2).label('qualidade_media')
        ).group_by(ExpertEvaluation.people_count).all()
        
        # Analysis by context
        por_contexto = db.session.query(
            ExpertEvaluation.photo_context,
            db.func.count().label('quantidade'),
            db.func.round(db.func.avg(ExpertEvaluation.overall_quality), 2).label('qualidade_media')
        ).group_by(ExpertEvaluation.photo_context).order_by(
            db.func.avg(ExpertEvaluation.overall_quality).desc()
        ).all()
        
        return render_template('dashboard.html',
                             total_avaliacoes=total_avaliacoes,
                             media_qualidade=media_qualidade,
                             rejeitadas=rejeitadas,
                             media_confianca=media_confianca,
                             por_pessoas=por_pessoas,
                             por_contexto=por_contexto)
    
    except Exception as e:
        logger.error(f"Error generating dashboard: {e}")
        return render_template('dashboard.html', 
                             total_avaliacoes=0,
                             media_qualidade=0,
                             rejeitadas=0,
                             media_confianca=0,
                             por_pessoas=[],
                             por_contexto=[])

@app.route('/export')
def export_data():
    """Export evaluation data as JSON"""
    if 'evaluator_id' not in session:
        return redirect(url_for('index'))
    
    try:
        evaluations = ExpertEvaluation.query.all()
        data = []
        
        for eval in evaluations:
            data.append({
                'image_filename': eval.image_filename,
                'evaluator_id': eval.evaluator_id,
                'timestamp': eval.timestamp.isoformat() if eval.timestamp else None,
                'overall_quality': eval.overall_quality,
                'global_sharpness': eval.global_sharpness,
                'person_sharpness': eval.person_sharpness,
                'exposure_quality': eval.exposure_quality,
                'composition_quality': eval.composition_quality,
                'emotional_impact': eval.emotional_impact,
                'technical_execution': eval.technical_execution,
                'environment_lighting': eval.environment_lighting,
                'person_lighting': eval.person_lighting,
                'person_sharpness_level': eval.person_sharpness_level,
                'person_position': eval.person_position,
                'eyes_quality': eval.eyes_quality,
                'people_count': eval.people_count,
                'photo_context': eval.photo_context,
                'technical_issues': eval.technical_issues,
                'approve_for_portfolio': eval.approve_for_portfolio,
                'approve_for_client': eval.approve_for_client,
                'approve_for_social': eval.approve_for_social,
                'needs_editing': eval.needs_editing,
                'complete_reject': eval.complete_reject,
                'confidence_level': eval.confidence_level,
                'evaluation_time_seconds': eval.evaluation_time_seconds,
                'comments': eval.comments
            })
        
        # Create temporary JSON file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            temp_path = f.name
        
        return send_file(temp_path, 
                        as_attachment=True, 
                        download_name=f'expert_evaluations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                        mimetype='application/json')
    
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return jsonify({'error': 'Failed to export data'}), 500

@app.route('/logout')
def logout():
    """Logout and clear session"""
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Create tables
    with app.app_context():
        db.create_all()
    
    # Run development server
    app.run(host='0.0.0.0', port=5001, debug=True)
