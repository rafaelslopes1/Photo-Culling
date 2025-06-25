#!/usr/bin/env python3
"""
Face Clustering System - Sistema de Agrupamento de Rostos
Implementa clustering de pessoas únicas usando face encodings
"""

import os
import sys
import json
import sqlite3
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import cv2
from sklearn.cluster import DBSCAN
from collections import defaultdict, Counter
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.face_recognition_system import FaceRecognitionSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceClusteringSystem:
    """
    Sistema de clustering de rostos para identificação de pessoas únicas
    """
    
    def __init__(self, features_db_path: str = "data/features/features.db"):
        """
        Initialize face clustering system
        
        Args:
            features_db_path: Path to features database
        """
        self.features_db_path = features_db_path
        self.face_recognition_system = None
        self.clusters_cache = {}
        self.person_names = {}  # cluster_id -> person_name mapping
        
        # Initialize database
        self._init_face_database()
        
    def _init_face_database(self):
        """
        Initialize face encodings database
        """
        try:
            conn = sqlite3.connect(self.features_db_path)
            cursor = conn.cursor()
            
            # Create face_encodings table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_encodings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    face_id INTEGER NOT NULL,
                    encoding BLOB NOT NULL,
                    bbox_x INTEGER,
                    bbox_y INTEGER,
                    bbox_w INTEGER,
                    bbox_h INTEGER,
                    confidence REAL,
                    extracted_at TEXT,
                    FOREIGN KEY (filename) REFERENCES image_features (filename)
                )
            ''')
            
            # Create person_clusters table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS person_clusters (
                    cluster_id INTEGER PRIMARY KEY,
                    person_name TEXT,
                    face_count INTEGER DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            
            # Create face_cluster_assignments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_cluster_assignments (
                    encoding_id INTEGER,
                    cluster_id INTEGER,
                    distance REAL,
                    assigned_at TEXT,
                    FOREIGN KEY (encoding_id) REFERENCES face_encodings (id),
                    FOREIGN KEY (cluster_id) REFERENCES person_clusters (cluster_id),
                    PRIMARY KEY (encoding_id, cluster_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Face database initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing face database: {e}")
            raise
    
    def extract_and_store_face_encodings(self, image_paths: List[str], force_recompute: bool = False) -> Dict:
        """
        Extract face encodings from images and store in database
        
        Args:
            image_paths: List of image file paths
            force_recompute: If True, recompute encodings even if they exist
            
        Returns:
            Dict with extraction statistics
        """
        if not self.face_recognition_system:
            self.face_recognition_system = FaceRecognitionSystem()
        
        stats = {
            'total_images': len(image_paths),
            'images_processed': 0,
            'faces_detected': 0,
            'encodings_extracted': 0,
            'errors': 0,
            'skipped': 0
        }
        
        conn = sqlite3.connect(self.features_db_path)
        cursor = conn.cursor()
        
        for image_path in image_paths:
            try:
                filename = os.path.basename(image_path)
                
                # Check if encodings already exist
                if not force_recompute:
                    cursor.execute('SELECT COUNT(*) FROM face_encodings WHERE filename = ?', (filename,))
                    existing_count = cursor.fetchone()[0]
                    if existing_count > 0:
                        stats['skipped'] += 1
                        continue
                
                # Load image and extract face encodings
                face_encodings = self.face_recognition_system.extract_face_encoding(image_path)
                
                if face_encodings:
                    stats['images_processed'] += 1
                    stats['faces_detected'] += len(face_encodings)
                    
                    for i, face_encoding in enumerate(face_encodings):
                        if face_encoding.face_encoding is not None:
                            # Store encoding in database
                            encoding_blob = pickle.dumps(face_encoding.face_encoding)
                            bbox = face_encoding.face_bbox
                            
                            cursor.execute('''
                                INSERT INTO face_encodings 
                                (filename, face_id, encoding, bbox_x, bbox_y, bbox_w, bbox_h, 
                                 confidence, extracted_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (
                                filename, i, encoding_blob,
                                bbox[0], bbox[1], bbox[2], bbox[3],
                                face_encoding.confidence,
                                datetime.now().isoformat()
                            ))
                            
                            stats['encodings_extracted'] += 1
                else:
                    logger.debug(f"No faces detected in: {filename}")
                    stats['images_processed'] += 1
                    
            except Exception as e:
                logger.error(f"❌ Error processing {image_path}: {e}")
                stats['errors'] += 1
                continue
        
        conn.commit()
        conn.close()
        
        logger.info(f"📊 Face extraction completed:")
        logger.info(f"   Images processed: {stats['images_processed']}/{stats['total_images']}")
        logger.info(f"   Faces detected: {stats['faces_detected']}")
        logger.info(f"   Encodings extracted: {stats['encodings_extracted']}")
        logger.info(f"   Errors: {stats['errors']}")
        logger.info(f"   Skipped: {stats['skipped']}")
        
        return stats
    
    def cluster_faces(self, eps: float = 0.5, min_samples: int = 2) -> Dict:
        """
        Cluster face encodings to identify unique persons
        
        Args:
            eps: DBSCAN epsilon parameter (distance threshold)
            min_samples: Minimum samples per cluster
            
        Returns:
            Dict with clustering results
        """
        logger.info(f"🔄 Starting face clustering with eps={eps}, min_samples={min_samples}")
        
        conn = sqlite3.connect(self.features_db_path)
        cursor = conn.cursor()
        
        # Get all face encodings
        cursor.execute('SELECT id, filename, face_id, encoding FROM face_encodings')
        rows = cursor.fetchall()
        
        if not rows:
            logger.warning("⚠️ No face encodings found in database")
            return {'error': 'No face encodings found'}
        
        # Prepare data for clustering
        encoding_ids = []
        encodings = []
        filenames = []
        
        for row in rows:
            encoding_id, filename, face_id, encoding_blob = row
            try:
                encoding = pickle.loads(encoding_blob)
                encoding_ids.append(encoding_id)
                encodings.append(encoding)
                filenames.append(filename)
            except Exception as e:
                logger.warning(f"⚠️ Could not deserialize encoding for {filename}: {e}")
                continue
        
        if len(encodings) < 2:
            logger.warning("⚠️ Not enough valid encodings for clustering")
            return {'error': 'Not enough encodings for clustering'}
        
        # Perform clustering
        encodings_array = np.array(encodings)
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        cluster_labels = clustering.fit_predict(encodings_array)
        
        # Process clustering results
        cluster_stats = defaultdict(list)
        noise_count = 0
        
        for i, label in enumerate(cluster_labels):
            if label == -1:  # Noise/outlier
                noise_count += 1
            else:
                cluster_stats[label].append({
                    'encoding_id': encoding_ids[i],
                    'filename': filenames[i]
                })
        
        # Store clustering results
        timestamp = datetime.now().isoformat()
        
        # Clear previous clustering results
        cursor.execute('DELETE FROM person_clusters')
        cursor.execute('DELETE FROM face_cluster_assignments')
        
        # Store new clusters
        for cluster_id, faces in cluster_stats.items():
            cursor.execute('''
                INSERT INTO person_clusters (cluster_id, face_count, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            ''', (cluster_id, len(faces), timestamp, timestamp))
            
            # Store face assignments
            for face in faces:
                cursor.execute('''
                    INSERT INTO face_cluster_assignments (encoding_id, cluster_id, assigned_at)
                    VALUES (?, ?, ?)
                ''', (face['encoding_id'], cluster_id, timestamp))
        
        conn.commit()
        conn.close()
        
        # Prepare results
        results = {
            'total_faces': len(encodings),
            'total_clusters': len(cluster_stats),
            'noise_faces': noise_count,
            'cluster_distribution': {k: len(v) for k, v in cluster_stats.items()},
            'clustering_params': {'eps': eps, 'min_samples': min_samples},
            'timestamp': timestamp
        }
        
        logger.info(f"✅ Face clustering completed:")
        logger.info(f"   Total faces: {results['total_faces']}")
        logger.info(f"   Unique persons: {results['total_clusters']}")
        logger.info(f"   Noise/outliers: {results['noise_faces']}")
        logger.info(f"   Largest clusters: {sorted(results['cluster_distribution'].values(), reverse=True)[:5]}")
        
        return results
    
    def get_person_faces(self, cluster_id: int) -> List[Dict]:
        """
        Get all faces belonging to a specific person cluster
        
        Args:
            cluster_id: Cluster ID of the person
            
        Returns:
            List of face data dictionaries
        """
        conn = sqlite3.connect(self.features_db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT fe.filename, fe.face_id, fe.bbox_x, fe.bbox_y, fe.bbox_w, fe.bbox_h, 
                   fe.confidence, fca.distance
            FROM face_encodings fe
            JOIN face_cluster_assignments fca ON fe.id = fca.encoding_id
            WHERE fca.cluster_id = ?
            ORDER BY fe.confidence DESC
        '''
        
        cursor.execute(query, (cluster_id,))
        rows = cursor.fetchall()
        conn.close()
        
        faces = []
        for row in rows:
            faces.append({
                'filename': row[0],
                'face_id': row[1],
                'bbox': [row[2], row[3], row[4], row[5]],
                'confidence': row[6],
                'distance': row[7]
            })
        
        return faces
    
    def search_similar_faces(self, query_image_path: str, top_k: int = 10) -> List[Dict]:
        """
        Search for faces similar to faces in query image
        
        Args:
            query_image_path: Path to query image
            top_k: Number of similar faces to return
            
        Returns:
            List of similar faces with distances
        """
        if not self.face_recognition_system:
            self.face_recognition_system = FaceRecognitionSystem()
        
        # Extract face encoding from query image
        face_encodings = self.face_recognition_system.extract_face_encoding(query_image_path)
        
        if not face_encodings:
            return []
        
        query_encoding = face_encodings[0].face_encoding
        if query_encoding is None:
            return []
        
        # Get all stored encodings
        conn = sqlite3.connect(self.features_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT fe.id, fe.filename, fe.face_id, fe.encoding, 
                   fe.bbox_x, fe.bbox_y, fe.bbox_w, fe.bbox_h,
                   fca.cluster_id
            FROM face_encodings fe
            LEFT JOIN face_cluster_assignments fca ON fe.id = fca.encoding_id
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        # Calculate distances
        similar_faces = []
        for row in rows:
            try:
                stored_encoding = pickle.loads(row[3])
                distance = np.linalg.norm(query_encoding - stored_encoding)
                
                similar_faces.append({
                    'filename': row[1],
                    'face_id': row[2],
                    'bbox': [row[4], row[5], row[6], row[7]],
                    'cluster_id': row[8],
                    'distance': distance
                })
            except:
                continue
        
        # Sort by distance and return top_k
        similar_faces.sort(key=lambda x: x['distance'])
        return similar_faces[:top_k]
    
    def get_clustering_statistics(self) -> Dict:
        """
        Get comprehensive clustering statistics
        
        Returns:
            Dict with clustering statistics
        """
        conn = sqlite3.connect(self.features_db_path)
        cursor = conn.cursor()
        
        # Basic stats
        cursor.execute('SELECT COUNT(*) FROM face_encodings')
        total_faces = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM person_clusters')
        total_clusters = cursor.fetchone()[0]
        
        # Cluster distribution
        cursor.execute('''
            SELECT cluster_id, face_count 
            FROM person_clusters 
            ORDER BY face_count DESC
        ''')
        cluster_distribution = cursor.fetchall()
        
        # Faces without clusters (noise)
        cursor.execute('''
            SELECT COUNT(*) FROM face_encodings fe
            LEFT JOIN face_cluster_assignments fca ON fe.id = fca.encoding_id
            WHERE fca.cluster_id IS NULL
        ''')
        noise_faces = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_faces': total_faces,
            'total_clusters': total_clusters,  
            'noise_faces': noise_faces,
            'cluster_distribution': cluster_distribution,
            'average_faces_per_cluster': total_faces / total_clusters if total_clusters > 0 else 0
        }


if __name__ == "__main__":
    # Example usage
    clustering_system = FaceClusteringSystem()
    
    # Test with a few images
    test_images = [
        "data/input/IMG_0001.JPG",
        "data/input/IMG_0239.JPG", 
        "data/input/IMG_0243.JPG",
        "data/input/IMG_0244.JPG",
        "data/input/IMG_0252.JPG"
    ]
    
    # Extract face encodings
    logger.info("🔄 Extracting face encodings...")
    extraction_stats = clustering_system.extract_and_store_face_encodings(test_images)
    
    # Perform clustering
    if extraction_stats['encodings_extracted'] > 0:
        logger.info("🔄 Clustering faces...")
        clustering_results = clustering_system.cluster_faces()
        
        # Show statistics
        stats = clustering_system.get_clustering_statistics()
        logger.info(f"📊 Final statistics: {stats}")
    else:
        logger.warning("⚠️ No face encodings extracted, skipping clustering")
