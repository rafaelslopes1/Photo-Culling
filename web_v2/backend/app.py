from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import json
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

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

# Import only essential components for web app
try:
    from src.core.feature_extractor import FeatureExtractor
    from src.utils.config_manager import ConfigManager
    FEATURE_EXTRACTION_AVAILABLE = True
    logger.info("Feature extraction components loaded successfully")
except ImportError as e:
    logger.warning(f"Feature extraction not available: {e}")
    FEATURE_EXTRACTION_AVAILABLE = False

# Initialize components only if available
if FEATURE_EXTRACTION_AVAILABLE:
    feature_extractor = FeatureExtractor()
    config_manager = ConfigManager()
else:
    feature_extractor = None
    config_manager = None

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
    end_time = db.Column(db.DateTime)
    images_evaluated = db.Column(db.Integer, default=0)
    total_time_seconds = db.Column(db.Integer, default=0)
    session_notes = db.Column(db.Text)

# Helper Functions
class ImageBatchManager:
    """Manages batches of images for evaluation"""
    
    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
        
    def get_image_list(self):
        """Get list of all images available for evaluation"""
        images = []
        for file_path in self.image_dir.rglob('*'):
            if file_path.suffix.lower() in self.supported_formats:
                images.append({
                    'filename': file_path.name,
                    'path': str(file_path),
                    'relative_path': str(file_path.relative_to(self.image_dir.parent))
                })
        return sorted(images, key=lambda x: x['filename'])
    
    def get_unevaluated_images(self, evaluator_id, limit=None):
        """Get images that haven't been evaluated by this evaluator"""
        all_images = self.get_image_list()
        
        # Get already evaluated images
        evaluated = db.session.query(ExpertEvaluation.image_filename).filter_by(
            evaluator_id=evaluator_id
        ).all()
        evaluated_filenames = {row[0] for row in evaluated}
        
        # Filter out already evaluated
        unevaluated = [
            img for img in all_images 
            if img['filename'] not in evaluated_filenames
        ]
        
        if limit:
            unevaluated = unevaluated[:limit]
            
        return unevaluated
    
    def create_evaluation_batch(self, evaluator_id, batch_size=50):
        """Create intelligent batch for evaluation"""
        unevaluated = self.get_unevaluated_images(evaluator_id, batch_size)
        
        # Add technical features for context
        for img in unevaluated:
            try:
                features = feature_extractor.extract_features(img['path'])
                img['technical_preview'] = {
                    'sharpness': features.get('sharpness_laplacian', 0),
                    'brightness': features.get('brightness_mean', 0),
                    'person_count': features.get('total_persons', 0),
                    'face_count': features.get('face_count', 0)
                }
            except Exception as e:
                logger.warning(f"Could not extract features for {img['filename']}: {e}")
                img['technical_preview'] = {}
        
        return unevaluated

# Initialize batch manager
image_batch_manager = ImageBatchManager(project_root / 'data' / 'input')

# Routes
@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Expert login"""
    if request.method == 'POST':
        evaluator_id = request.form.get('evaluator_id')
        if evaluator_id:
            session['evaluator_id'] = evaluator_id
            
            # Create or update session record
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
    
    # Get batch of images for evaluation
    batch = image_batch_manager.create_evaluation_batch(
        session['evaluator_id'], batch_size=50
    )
    
    if not batch:
        return render_template('completed.html', message="All images evaluated!")
    
    # Get current progress
    total_images = len(image_batch_manager.get_image_list())
    evaluated_count = db.session.query(ExpertEvaluation).filter_by(
        evaluator_id=session['evaluator_id']
    ).count()
    
    progress = {
        'evaluated': evaluated_count,
        'total': total_images,
        'remaining': len(batch),
        'percentage': round((evaluated_count / total_images) * 100, 1) if total_images > 0 else 0
    }
    
    return render_template('evaluate_v2.html', 
                         images=batch, 
                         progress=progress,
                         evaluator_id=session['evaluator_id'])

@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    """API endpoint to save evaluation"""
    if 'evaluator_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.json
    
    # Validate input data
    if not data:
        return jsonify({'error': 'No data provided'}), 400
        
    if 'image_filename' not in data:
        return jsonify({'error': 'Image filename required'}), 400
    
    try:
        # Extract data with defaults
        ratings = data.get('ratings', {})
        decisions = data.get('decisions', {})
        categorical = data.get('categorical_assessments', {})
        
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
        
        return jsonify({'success': True, 'evaluation_id': evaluation.id})
        
    except Exception as e:
        logger.error(f"Error saving evaluation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/progress')
def api_progress():
    """Get current evaluation progress"""
    if 'evaluator_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    total_images = len(image_batch_manager.get_image_list())
    evaluated_count = db.session.query(ExpertEvaluation).filter_by(
        evaluator_id=session['evaluator_id']
    ).count()
    
    return jsonify({
        'evaluated': evaluated_count,
        'total': total_images,
        'remaining': total_images - evaluated_count,
        'percentage': round((evaluated_count / total_images) * 100, 1) if total_images > 0 else 0
    })

@app.route('/analytics')
def analytics():
    """Analytics dashboard"""
    if 'evaluator_id' not in session:
        return redirect(url_for('login'))
    
    # Get evaluation statistics
    evaluations = ExpertEvaluation.query.filter_by(
        evaluator_id=session['evaluator_id']
    ).all()
    
    stats = {
        'total_evaluations': len(evaluations),
        'average_ratings': {},
        'decision_distribution': {},
        'evaluation_speed': 0
    }
    
    if evaluations:
        # Calculate average ratings
        rating_fields = ['overall_quality', 'global_sharpness', 'person_sharpness', 
                        'exposure_quality', 'composition_quality', 'emotional_impact', 
                        'technical_execution']
        
        for field in rating_fields:
            values = [getattr(eval, field) for eval in evaluations if getattr(eval, field)]
            stats['average_ratings'][field] = round(sum(values) / len(values), 2) if values else 0
        
        # Decision distribution
        decision_fields = ['approve_for_portfolio', 'approve_for_client', 'approve_for_social',
                          'needs_editing', 'complete_reject']
        
        for field in decision_fields:
            true_count = sum(1 for eval in evaluations if getattr(eval, field))
            stats['decision_distribution'][field] = {
                'count': true_count,
                'percentage': round((true_count / len(evaluations)) * 100, 1)
            }
        
        # Average evaluation speed
        times = [eval.evaluation_time_seconds for eval in evaluations if eval.evaluation_time_seconds]
        stats['evaluation_speed'] = round(sum(times) / len(times), 1) if times else 0
    
    return render_template('analytics.html', stats=stats)

@app.route('/export')
def export_evaluations():
    """Export evaluations as JSON"""
    if 'evaluator_id' not in session:
        return redirect(url_for('login'))
    
    evaluations = ExpertEvaluation.query.filter_by(
        evaluator_id=session['evaluator_id']
    ).all()
    
    export_data = {
        'evaluator_id': session['evaluator_id'],
        'export_timestamp': datetime.utcnow().isoformat(),
        'total_evaluations': len(evaluations),
        'evaluations': [eval.to_dict() for eval in evaluations]
    }
    
    return jsonify(export_data)

# API Routes for image handling
@app.route('/api/images')
def get_images():
    """Get list of available images for evaluation"""
    try:
        # Get images from data/input directory
        input_dir = project_root / 'data' / 'input'
        if not input_dir.exists():
            return jsonify({'success': False, 'error': 'Input directory not found'})
        
        # Get image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        images = []
        
        for file_path in input_dir.iterdir():
            if file_path.suffix.lower() in image_extensions:
                images.append({
                    'filename': file_path.name,
                    'path': str(file_path),
                    'size': file_path.stat().st_size
                })
        
        # Sort by filename
        images.sort(key=lambda x: x['filename'])
        
        logger.info(f"Found {len(images)} images for evaluation")
        return jsonify({'success': True, 'images': images})
        
    except Exception as e:
        logger.error(f"Error getting images: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/image/<filename>')
def serve_image(filename):
    """Serve image file for evaluation"""
    try:
        input_dir = project_root / 'data' / 'input'
        image_path = input_dir / filename
        
        if not image_path.exists():
            return "Image not found", 404
            
        return send_file(str(image_path))
        
    except Exception as e:
        logger.error(f"Error serving image {filename}: {e}")
        return "Error serving image", 500

# Initialize database
def init_database():
    """Create database tables"""
    with app.app_context():
        db.create_all()
        logger.info("Database tables created")

if __name__ == '__main__':
    # Create database tables
    with app.app_context():
        db.create_all()
        logger.info("Database initialized")
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5001)
