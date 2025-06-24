#!/usr/bin/env python3
"""
Face Recognition and Clustering Module - Photo Culling System v2.5
Módulo de reconhecimento e clustering facial

Implements Phase 3: Face recognition and clustering using face_recognition library
with MediaPipe integration for robust face detection and clustering.
"""

import os
import cv2
import json
import numpy as np
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
from datetime import datetime
import pickle
import base64
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
import face_recognition

logger = logging.getLogger(__name__)


@dataclass
class FaceEncoding:
    """Face encoding data structure with face_recognition integration"""
    face_id: str
    image_filename: str
    face_bbox: Tuple[int, int, int, int]  # x, y, w, h
    landmarks: List[Tuple[float, float]]
    confidence: float
    face_encoding: np.ndarray  # 128-dimensional face encoding from face_recognition
    encoding_base64: str  # Base64 encoded for database storage
    timestamp: str


@dataclass 
class FaceCluster:
    """Face cluster data structure"""
    cluster_id: str
    representative_face_id: str
    face_ids: List[str]
    cluster_center: Dict[str, float]
    cluster_size: int
    confidence_avg: float
    image_count: int
    created_timestamp: str
    updated_timestamp: str


class FaceRecognitionSystem:
    """
    Face recognition and clustering system using MediaPipe
    Sistema de reconhecimento e clustering facial usando MediaPipe
    """
    
    def __init__(self, database_path: str = "data/features/face_recognition.db"):
        """Initialize face recognition system"""
        self.database_path = database_path
        self.database_dir = Path(database_path).parent
        self.database_dir.mkdir(parents=True, exist_ok=True)
        
        # Import required modules
        try:
            from .person_detector import PersonDetector
            self.person_detector = PersonDetector()
        except ImportError:
            from src.core.person_detector import PersonDetector
            self.person_detector = PersonDetector()
        
        # Initialize database
        self._init_database()
        
        # Similarity thresholds for face_recognition
        self.similarity_threshold = 0.6  # Standard threshold for face_recognition
        self.min_cluster_size = 2
        self.max_cluster_size = 100
        
        logger.info("FaceRecognitionSystem inicializado com face_recognition")
    
    def _init_database(self):
        """Initialize SQLite database for face recognition"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Face encodings table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS face_encodings (
                        face_id TEXT PRIMARY KEY,
                        image_filename TEXT NOT NULL,
                        face_bbox TEXT NOT NULL,
                        landmarks TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        face_encoding_base64 TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        UNIQUE(image_filename, face_bbox)
                    )
                """)
                
                # Face clusters table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS face_clusters (
                        cluster_id TEXT PRIMARY KEY,
                        representative_face_id TEXT NOT NULL,
                        face_ids TEXT NOT NULL,
                        cluster_center TEXT NOT NULL,
                        cluster_size INTEGER NOT NULL,
                        confidence_avg REAL NOT NULL,
                        image_count INTEGER NOT NULL,
                        created_timestamp TEXT NOT NULL,
                        updated_timestamp TEXT NOT NULL
                    )
                """)
                
                # Face cluster assignments table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS face_cluster_assignments (
                        face_id TEXT NOT NULL,
                        cluster_id TEXT NOT NULL,
                        similarity_score REAL NOT NULL,
                        assignment_timestamp TEXT NOT NULL,
                        PRIMARY KEY (face_id, cluster_id)
                    )
                """)
                
                conn.commit()
                logger.info("Base de dados de reconhecimento facial inicializada")
                
        except Exception as e:
            logger.error(f"Erro ao inicializar base de dados: {e}")
            raise
    
    def extract_face_encoding(self, image_path: str) -> List[FaceEncoding]:
        """Extract face encodings using MediaPipe for detection + face_recognition for encoding"""
        try:
            # First, use MediaPipe (via PersonDetector) to detect faces
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
            
            # Detect faces using MediaPipe
            result = self.person_detector.detect_persons_and_faces(image)
            faces = result.get('faces', [])
            
            if not faces:
                return []
            
            # Convert to RGB for face_recognition
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            encodings = []
            for i, face in enumerate(faces):
                try:
                    # Get face bounding box from MediaPipe
                    bbox = face.get('bbox', [0, 0, 1, 1])
                    x, y, w, h = bbox
                    
                    # Convert to face_recognition format (top, right, bottom, left)
                    face_location = (y, x + w, y + h, x)
                    
                    # Extract encoding using face_recognition
                    face_encodings = face_recognition.face_encodings(rgb_image, [face_location])
                    
                    if face_encodings:
                        face_encoding = face_encodings[0]
                        
                        # Get landmarks if available
                        landmarks = face.get('landmarks', [])
                        
                        # Create face encoding
                        face_id = f"{Path(image_path).stem}_face_{i}_{int(datetime.now().timestamp())}"
                        
                        # Convert encoding to base64 for storage
                        encoding_base64 = base64.b64encode(face_encoding.tobytes()).decode('utf-8')
                        
                        encoding = FaceEncoding(
                            face_id=face_id,
                            image_filename=Path(image_path).name,
                            face_bbox=bbox,
                            landmarks=landmarks,
                            confidence=face.get('confidence', 1.0),
                            face_encoding=face_encoding,
                            encoding_base64=encoding_base64,
                            timestamp=datetime.now().isoformat()
                        )
                        
                        encodings.append(encoding)
                        
                    else:
                        logger.warning(f"face_recognition falhou em extrair encoding da face {i} em {image_path}")
                        
                except Exception as e:
                    logger.warning(f"Erro ao processar face {i} em {image_path}: {e}")
                    continue
            
            return encodings
            
        except Exception as e:
            logger.error(f"Erro ao extrair encoding facial de {image_path}: {e}")
            return []
    
    def calculate_face_similarity(self, encoding1: FaceEncoding, encoding2: FaceEncoding) -> float:
        """Calculate similarity between two face encodings using face_recognition distance"""
        try:
            # Use face_recognition's built-in distance calculation
            distance = face_recognition.face_distance([encoding1.face_encoding], encoding2.face_encoding)[0]
            
            # Convert distance to similarity (0 = same face, 1 = different faces)
            # face_recognition distance: 0.0 = identical, 0.6 = threshold, higher = more different
            similarity = max(0.0, 1.0 - (distance / 0.6))  # Normalize to 0-1 range
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Erro ao calcular similaridade: {e}")
            return 0.0
    
    def store_face_encoding(self, encoding: FaceEncoding) -> bool:
        """Store face encoding in database"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO face_encodings 
                    (face_id, image_filename, face_bbox, landmarks, confidence, 
                     face_encoding_base64, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    encoding.face_id,
                    encoding.image_filename, 
                    json.dumps(encoding.face_bbox),
                    json.dumps(encoding.landmarks),
                    encoding.confidence,
                    encoding.encoding_base64,
                    encoding.timestamp
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Erro ao armazenar encoding: {e}")
            return False
    
    def get_all_face_encodings(self) -> List[FaceEncoding]:
        """Retrieve all face encodings from database"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM face_encodings")
                rows = cursor.fetchall()
                
                encodings = []
                for row in rows:
                    # Decode the face encoding from base64
                    face_encoding_bytes = base64.b64decode(row[5])
                    face_encoding = np.frombuffer(face_encoding_bytes, dtype=np.float64)
                    
                    encoding = FaceEncoding(
                        face_id=row[0],
                        image_filename=row[1],
                        face_bbox=tuple(json.loads(row[2])),
                        landmarks=json.loads(row[3]),
                        confidence=row[4],
                        face_encoding=face_encoding,
                        encoding_base64=row[5],
                        timestamp=row[6]
                    )
                    encodings.append(encoding)
                
                return encodings
                
        except Exception as e:
            logger.error(f"Erro ao recuperar encodings: {e}")
            return []
    
    def find_similar_faces(self, target_encoding: FaceEncoding, 
                          threshold: Optional[float] = None) -> List[Tuple[FaceEncoding, float]]:
        """Find faces similar to target encoding"""
        if threshold is None:
            threshold = self.similarity_threshold
        
        all_encodings = self.get_all_face_encodings()
        similar_faces = []
        
        for encoding in all_encodings:
            if encoding.face_id != target_encoding.face_id:
                similarity = self.calculate_face_similarity(target_encoding, encoding)
                if similarity >= threshold:
                    similar_faces.append((encoding, similarity))
        
        # Sort by similarity (highest first)
        similar_faces.sort(key=lambda x: x[1], reverse=True)
        return similar_faces
    
    def cluster_faces(self, eps: float = 0.6, min_samples: int = 2) -> List[FaceCluster]:
        """Cluster faces using DBSCAN with face_recognition encodings"""
        try:
            # Get all face encodings
            all_encodings = self.get_all_face_encodings()
            
            if len(all_encodings) < 2:
                return []
            
            # Prepare data for clustering
            face_encodings_matrix = np.array([enc.face_encoding for enc in all_encodings])
            
            # Use DBSCAN clustering
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            cluster_labels = clustering.fit_predict(face_encodings_matrix)
            
            # Group faces by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label != -1:  # Ignore noise points
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(all_encodings[i])
            
            # Create FaceCluster objects
            face_clusters = []
            for cluster_id, encodings in clusters.items():
                if len(encodings) >= min_samples:
                    # Select representative face (highest confidence)
                    representative = max(encodings, key=lambda x: x.confidence)
                    
                    cluster = FaceCluster(
                        cluster_id=f"cluster_{cluster_id}_{int(datetime.now().timestamp())}",
                        representative_face_id=representative.face_id,
                        face_ids=[enc.face_id for enc in encodings],
                        cluster_center=self._calculate_cluster_center(encodings),
                        cluster_size=len(encodings),
                        confidence_avg=float(np.mean([enc.confidence for enc in encodings])),
                        image_count=len(set(enc.image_filename for enc in encodings)),
                        created_timestamp=datetime.now().isoformat(),
                        updated_timestamp=datetime.now().isoformat()
                    )
                    
                    face_clusters.append(cluster)
                    
                    # Store cluster in database
                    self.store_face_cluster(cluster)
            
            logger.info(f"Criados {len(face_clusters)} clusters de faces")
            return face_clusters
            
        except Exception as e:
            logger.error(f"Erro no clustering de faces: {e}")
            return []

    def process_image_for_face_recognition(self, image_path: str) -> Dict:
        """Process single image for face recognition"""
        try:
            # Extract face encodings
            encodings = self.extract_face_encoding(image_path)
            
            if not encodings:
                return {
                    'status': 'no_faces',
                    'faces_found': 0,
                    'similar_faces': [],
                    'image_path': image_path
                }
            
            # Store encodings
            stored_count = 0
            similar_faces = []
            
            for encoding in encodings:
                if self.store_face_encoding(encoding):
                    stored_count += 1
                    
                    # Find similar faces
                    similar = self.find_similar_faces(encoding)
                    if similar:
                        similar_faces.extend([(face.face_id, score) for face, score in similar])
            
            return {
                'status': 'success',
                'faces_found': len(encodings),
                'faces_stored': stored_count,
                'similar_faces': similar_faces,
                'image_path': image_path
            }
            
        except Exception as e:
            logger.error(f"Erro no processamento de {image_path}: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'faces_found': 0,
                'similar_faces': [],
                'image_path': image_path
            }

    def generate_face_recognition_report(self) -> Dict:
        """Generate comprehensive face recognition report"""
        try:
            all_encodings = self.get_all_face_encodings()
            all_clusters = self.get_face_clusters()
            
            # Basic statistics
            total_faces = len(all_encodings)
            total_images = len(set(enc.image_filename for enc in all_encodings))
            total_clusters = len(all_clusters)
            
            # Cluster statistics
            cluster_sizes = [cluster.cluster_size for cluster in all_clusters]
            avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
            largest_cluster = max(cluster_sizes) if cluster_sizes else 0
            
            # Confidence statistics
            confidences = [enc.confidence for enc in all_encodings]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            # Images with multiple faces
            image_face_counts = {}
            for enc in all_encodings:
                image_face_counts[enc.image_filename] = image_face_counts.get(enc.image_filename, 0) + 1
            
            images_with_multiple_faces = sum(1 for count in image_face_counts.values() if count > 1)
            
            return {
                'face_statistics': {
                    'total_faces': total_faces,
                    'total_images': total_images,
                    'avg_faces_per_image': total_faces / max(total_images, 1),
                    'images_with_multiple_faces': images_with_multiple_faces,
                    'avg_confidence': float(avg_confidence)
                },
                'cluster_statistics': {
                    'total_clusters': total_clusters,
                    'avg_cluster_size': float(avg_cluster_size),
                    'largest_cluster_size': largest_cluster,
                    'clustered_faces': sum(cluster_sizes),
                    'clustering_efficiency': sum(cluster_sizes) / max(total_faces, 1)
                },
                'top_clusters': sorted(all_clusters, key=lambda x: x.cluster_size, reverse=True)[:5],
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {e}")
            return {
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }
    
    def cleanup_database(self):
        """Clean up orphaned records and optimize database"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Remove assignments for non-existent faces
                cursor.execute("""
                    DELETE FROM face_cluster_assignments 
                    WHERE face_id NOT IN (SELECT face_id FROM face_encodings)
                """)
                
                # Remove assignments for non-existent clusters
                cursor.execute("""
                    DELETE FROM face_cluster_assignments
                    WHERE cluster_id NOT IN (SELECT cluster_id FROM face_clusters)
                """)
                
                # Vacuum database
                cursor.execute("VACUUM")
                
                conn.commit()
                logger.info("Limpeza da base de dados concluída")
                
        except Exception as e:
            logger.error(f"Erro na limpeza da base de dados: {e}")

    def _calculate_cluster_center(self, encodings: List[FaceEncoding]) -> Dict[str, float]:
        """Calculate cluster center from list of face encodings"""
        if not encodings:
            return {}
        
        try:
            # Stack all face encodings and calculate mean
            face_encodings = np.array([enc.face_encoding for enc in encodings])
            center_encoding = np.mean(face_encodings, axis=0)
            
            # Convert to dict for storage (just first 10 dimensions for space efficiency)
            cluster_center = {
                f'dim_{i}': float(center_encoding[i]) 
                for i in range(min(10, len(center_encoding)))
            }
            
            # Add some basic statistics
            cluster_center.update({
                'avg_confidence': float(np.mean([enc.confidence for enc in encodings])),
                'encoding_count': len(encodings),
                'center_norm': float(np.linalg.norm(center_encoding))
            })
            
            return cluster_center
            
        except Exception as e:
            logger.error(f"Erro ao calcular centro do cluster: {e}")
            return {'error': 1.0}

    def store_face_cluster(self, cluster: FaceCluster) -> bool:
        """Store face cluster in database"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Store cluster
                cursor.execute("""
                    INSERT OR REPLACE INTO face_clusters
                    (cluster_id, representative_face_id, face_ids, cluster_center,
                     cluster_size, confidence_avg, image_count, created_timestamp, updated_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    cluster.cluster_id,
                    cluster.representative_face_id,
                    json.dumps(cluster.face_ids),
                    json.dumps(cluster.cluster_center),
                    cluster.cluster_size,
                    cluster.confidence_avg,
                    cluster.image_count,
                    cluster.created_timestamp,
                    cluster.updated_timestamp
                ))
                
                # Store assignments
                for face_id in cluster.face_ids:
                    cursor.execute("""
                        INSERT OR REPLACE INTO face_cluster_assignments
                        (face_id, cluster_id, similarity_score, assignment_timestamp)
                        VALUES (?, ?, ?, ?)
                    """, (
                        face_id,
                        cluster.cluster_id,
                        1.0,  # Default similarity
                        datetime.now().isoformat()
                    ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Erro ao armazenar cluster: {e}")
            return False

    def get_face_clusters(self) -> List[FaceCluster]:
        """Get all face clusters from database"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM face_clusters")
                rows = cursor.fetchall()
                
                clusters = []
                for row in rows:
                    cluster = FaceCluster(
                        cluster_id=row[0],
                        representative_face_id=row[1],
                        face_ids=json.loads(row[2]),
                        cluster_center=json.loads(row[3]),
                        cluster_size=row[4],
                        confidence_avg=row[5],
                        image_count=row[6],
                        created_timestamp=row[7],
                        updated_timestamp=row[8]
                    )
                    clusters.append(cluster)
                
                return clusters
                
        except Exception as e:
            logger.error(f"Erro ao recuperar clusters: {e}")
            return []


def main():
    """Test function for face recognition system"""
    print("🧠 SISTEMA DE RECONHECIMENTO FACIAL - TESTE")
    
    # Initialize system
    face_system = FaceRecognitionSystem()
    
    # Test with a few images
    test_images = ["data/input/IMG_0001.JPG", "data/input/IMG_0239.JPG"]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"Processando: {image_path}")
            result = face_system.process_image_for_face_recognition(image_path)
            print(f"Resultado: {result['status']}, {result['faces_found']} faces")
    
    # Generate clusters
    print("Criando clusters...")
    clusters = face_system.cluster_faces()
    print(f"Criados {len(clusters)} clusters")
    
    # Generate report
    report = face_system.generate_face_recognition_report()
    print(f"Relatório: {report['face_statistics']['total_faces']} faces em {report['cluster_statistics']['total_clusters']} clusters")


if __name__ == "__main__":
    main()
