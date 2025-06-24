#!/usr/bin/env python3
"""
Face Recognition System Test - Photo Culling System v2.5
Teste do sistema de reconhecimento facial

Tests the complete face recognition and clustering pipeline with face_recognition library
"""

import os
import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.face_recognition_system import FaceRecognitionSystem
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_face_recognition_system():
    """Test the face recognition system with sample images"""
    
    print("🧠 TESTE DO SISTEMA DE RECONHECIMENTO FACIAL v2.5")
    print("=" * 60)
    
    # Initialize system
    print("Inicializando sistema de reconhecimento facial...")
    face_system = FaceRecognitionSystem()
    
    # Get sample images
    input_dir = Path("data/input")
    if not input_dir.exists():
        print("❌ Diretório data/input não encontrado")
        return
    
    # Get specific images that we know have people from previous tests
    test_image_names = [
        "IMG_0001.JPG", "IMG_0239.JPG", "IMG_0243.JPG", "IMG_0244.JPG", 
        "IMG_0277.JPG", "IMG_0285.JPG", "IMG_0304.JPG", "IMG_0334.JPG",
        "IMG_0339.JPG", "IMG_0343.JPG"
    ]
    
    image_files = []
    for name in test_image_names:
        path = input_dir / name
        if path.exists():
            image_files.append(path)
    
    if not image_files:
        print("❌ Nenhuma das imagens de teste encontrada em data/input")
        return
    
    print(f"Processando {len(image_files)} imagens de teste...")
    print("-" * 40)
    
    # Process images
    total_faces = 0
    successful_images = 0
    processing_times = []
    
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i:2d}/{len(image_files)}] {image_path.name}")
        
        start_time = time.time()
        result = face_system.process_image_for_face_recognition(str(image_path))
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        if result['status'] == 'success':
            successful_images += 1
            faces_found = result['faces_found']
            total_faces += faces_found
            similar_faces = len(result['similar_faces'])
            
            print(f"    ✅ {faces_found} face(s) detectada(s), {similar_faces} similar(es)")
            print(f"    ⏱️  Tempo: {processing_time:.2f}s")
            
        elif result['status'] == 'no_faces':
            print(f"    ⚪ Nenhuma face detectada")
            print(f"    ⏱️  Tempo: {processing_time:.2f}s")
            
        else:
            print(f"    ❌ Erro: {result.get('error', 'Desconhecido')}")
    
    print("-" * 40)
    print(f"📊 ESTATÍSTICAS DE PROCESSAMENTO:")
    print(f"   • Imagens processadas: {len(image_files)}")
    print(f"   • Imagens com sucesso: {successful_images}")
    print(f"   • Total de faces detectadas: {total_faces}")
    print(f"   • Média de faces por imagem: {total_faces/max(len(image_files), 1):.1f}")
    print(f"   • Tempo médio por imagem: {sum(processing_times)/len(processing_times):.2f}s")
    print(f"   • Velocidade: {len(image_files)/sum(processing_times):.1f} imagens/s")
    
    # Test clustering
    clusters = []
    if total_faces >= 2:
        print("\n🔄 TESTANDO CLUSTERING FACIAL...")
        print("-" * 40)
        
        start_time = time.time()
        clusters = face_system.cluster_faces()
        clustering_time = time.time() - start_time
        
        print(f"✅ Clustering concluído em {clustering_time:.2f}s")
        print(f"📊 {len(clusters)} clusters criados")
        
        if clusters:
            # Show cluster details
            for i, cluster in enumerate(clusters[:5], 1):  # Show first 5 clusters
                print(f"   Cluster {i}: {cluster.cluster_size} faces, confiança média: {cluster.confidence_avg:.2f}")
    
    # Generate comprehensive report
    print("\n📋 RELATÓRIO DETALHADO:")
    print("-" * 40)
    
    report = face_system.generate_face_recognition_report()
    
    if 'error' not in report:
        face_stats = report['face_statistics']
        cluster_stats = report['cluster_statistics']
        
        print(f"🧑 Estatísticas de Faces:")
        print(f"   • Total de faces: {face_stats['total_faces']}")
        print(f"   • Imagens com faces: {face_stats['total_images']}")
        print(f"   • Faces por imagem: {face_stats['avg_faces_per_image']:.1f}")
        print(f"   • Imagens com múltiplas faces: {face_stats['images_with_multiple_faces']}")
        print(f"   • Confiança média: {face_stats['avg_confidence']:.2f}")
        
        print(f"\n🎯 Estatísticas de Clustering:")
        print(f"   • Total de clusters: {cluster_stats['total_clusters']}")
        print(f"   • Tamanho médio do cluster: {cluster_stats['avg_cluster_size']:.1f}")
        print(f"   • Maior cluster: {cluster_stats['largest_cluster_size']} faces")
        print(f"   • Faces clusterizadas: {cluster_stats['clustered_faces']}")
        print(f"   • Eficiência do clustering: {cluster_stats['clustering_efficiency']:.1%}")
        
        # Show top clusters
        if report['top_clusters']:
            print(f"\n🏆 Top 3 Clusters:")
            for i, cluster in enumerate(report['top_clusters'][:3], 1):
                print(f"   {i}. Cluster {cluster.cluster_id[-8:]}: {cluster.cluster_size} faces ({cluster.image_count} imagens)")
    
    else:
        print(f"❌ Erro no relatório: {report['error']}")
    
    print("\n" + "=" * 60)
    print("✅ TESTE CONCLUÍDO!")
    
    # Save results
    results_dir = Path("data/quality/face_recognition_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    test_results = {
        'test_summary': {
            'images_processed': len(image_files),
            'successful_images': successful_images,
            'total_faces_detected': total_faces,
            'avg_processing_time': sum(processing_times) / len(processing_times),
            'processing_speed': len(image_files) / sum(processing_times),
            'clusters_created': len(clusters)
        },
        'detailed_report': report,
        'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(results_dir / "test_results.json", 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"📁 Resultados salvos em: {results_dir}/test_results.json")


if __name__ == "__main__":
    test_face_recognition_system()
