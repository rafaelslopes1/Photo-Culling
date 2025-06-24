#!/usr/bin/env python3
"""
Integrated Feature Extraction + Face Recognition Test
Teste integrado de extração de features + reconhecimento facial
"""

import os
import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.feature_extractor import FeatureExtractor
from src.core.face_recognition_system import FaceRecognitionSystem
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_integrated_system():
    """Test the integrated feature extraction + face recognition system"""
    
    print("🔗 TESTE INTEGRADO: EXTRAÇÃO DE FEATURES + RECONHECIMENTO FACIAL")
    print("=" * 70)
    
    # Initialize systems
    print("Inicializando sistemas...")
    feature_extractor = FeatureExtractor()
    face_system = FaceRecognitionSystem()
    
    # Get sample images
    input_dir = Path("data/input")
    test_images = ["IMG_0001.JPG", "IMG_0239.JPG", "IMG_0243.JPG", "IMG_0244.JPG", "IMG_0277.JPG"]
    
    print(f"Processando {len(test_images)} imagens de teste...")
    print("-" * 50)
    
    # Process images
    total_faces_fe = 0  # From feature extractor
    total_faces_fr = 0  # From face recognition
    processing_times = []
    
    for i, image_name in enumerate(test_images, 1):
        image_path = input_dir / image_name
        
        if not image_path.exists():
            continue
            
        print(f"[{i:2d}] {image_name}")
        
        start_time = time.time()
        
        # Extract features (includes face recognition)
        features = feature_extractor.extract_features(str(image_path))
        
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        # Analyze results
        face_count = features.get('face_count', 0)
        face_encodings_count = features.get('face_encodings_count', 0)
        similar_faces_count = features.get('similar_faces_count', 0)
        
        total_faces_fe += face_count
        total_faces_fr += face_encodings_count
        
        print(f"    📊 Features extraídas: {len(features)} características")
        print(f"    🧑 Faces MediaPipe: {face_count}")
        print(f"    🧠 Face encodings: {face_encodings_count}")
        print(f"    🔄 Faces similares: {similar_faces_count}")
        print(f"    ⏱️  Tempo: {processing_time:.2f}s")
        
        # Show some key features
        key_features = {
            'blur_score': features.get('sharpness_laplacian', 0),
            'brightness': features.get('brightness_mean', 0),
            'contrast': features.get('contrast_rms', 0),
            'person_count': features.get('person_count', 0)
        }
        print(f"    🔍 Qualidade: blur={key_features['blur_score']:.1f}, "
              f"brilho={key_features['brightness']:.1f}, "
              f"contraste={key_features['contrast']:.1f}, "
              f"pessoas={key_features['person_count']}")
    
    print("-" * 50)
    print(f"📊 ESTATÍSTICAS INTEGRADAS:")
    print(f"   • Imagens processadas: {len(test_images)}")
    print(f"   • Faces detectadas (MediaPipe): {total_faces_fe}")
    print(f"   • Face encodings (face_recognition): {total_faces_fr}")
    print(f"   • Taxa de sucesso encoding: {(total_faces_fr/max(total_faces_fe, 1))*100:.1f}%")
    print(f"   • Tempo médio por imagem: {sum(processing_times)/len(processing_times):.2f}s")
    print(f"   • Velocidade integrada: {len(test_images)/sum(processing_times):.1f} imagens/s")
    
    # Test clustering
    print(f"\n🔄 CLUSTERING DE FACES...")
    print("-" * 50)
    
    start_time = time.time()
    clusters = face_system.cluster_faces()
    clustering_time = time.time() - start_time
    
    print(f"✅ Clustering concluído em {clustering_time:.2f}s")
    print(f"📊 {len(clusters)} clusters criados")
    
    if clusters:
        for i, cluster in enumerate(clusters[:3], 1):  # Show first 3 clusters
            print(f"   Cluster {i}: {cluster.cluster_size} faces, "
                  f"confiança média: {cluster.confidence_avg:.2f}, "
                  f"{cluster.image_count} imagens")
    
    # Generate final report
    print(f"\n📋 RELATÓRIO FINAL:")
    print("-" * 50)
    
    report = face_system.generate_face_recognition_report()
    
    if 'error' not in report:
        face_stats = report['face_statistics']
        cluster_stats = report['cluster_statistics']
        
        print(f"🎯 Sistema Integrado Performance:")
        print(f"   • Pipeline completo: {sum(processing_times):.2f}s total")
        print(f"   • Features + Face Recognition: {sum(processing_times)/len(test_images):.2f}s/imagem")
        print(f"   • Clustering adicional: {clustering_time:.2f}s")
        print(f"   • Throughput integrado: {len(test_images)/(sum(processing_times) + clustering_time):.1f} imagens/s")
        
        print(f"\n🧠 Reconhecimento Facial:")
        print(f"   • Total de faces: {face_stats['total_faces']}")
        print(f"   • Clusters criados: {cluster_stats['total_clusters']}")
        print(f"   • Eficiência do clustering: {cluster_stats['clustering_efficiency']:.1%}")
        print(f"   • Faces por cluster: {cluster_stats['avg_cluster_size']:.1f}")
    
    # Save integrated results
    results_dir = Path("data/quality/integrated_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    integrated_results = {
        'integration_summary': {
            'images_processed': len(test_images),
            'faces_detected_mediapipe': total_faces_fe,
            'faces_encoded_face_recognition': total_faces_fr,
            'encoding_success_rate': (total_faces_fr/max(total_faces_fe, 1))*100,
            'avg_processing_time': sum(processing_times)/len(processing_times),
            'integrated_throughput': len(test_images)/(sum(processing_times) + clustering_time),
            'clusters_created': len(clusters)
        },
        'face_recognition_report': report,
        'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(results_dir / "integrated_results.json", 'w', encoding='utf-8') as f:
        json.dump(integrated_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📁 Resultados salvos em: {results_dir}/integrated_results.json")
    print("\n" + "=" * 70)
    print("✅ TESTE INTEGRADO CONCLUÍDO COM SUCESSO!")


if __name__ == "__main__":
    test_integrated_system()
