#!/usr/bin/env python3
"""
GPU Configuration and Optimization for Photo Culling System
Configuração e otimização de GPU para o sistema de seleção de fotos

Specialized for Mac M3 and MediaPipe optimization
"""

import os
import platform
import subprocess
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class MacM3Optimizer:
    """
    Specialized optimizer for Mac M3 systems
    Otimizador especializado para sistemas Mac M3
    """
    
    @staticmethod
    def detect_mac_silicon() -> Dict:
        """
        Detect Mac Silicon chip information
        Detectar informações do chip Mac Silicon
        """
        system_info = {
            'is_apple_silicon': False,
            'chip_type': 'unknown',
            'gpu_cores': 0,
            'cpu_cores': 0,
            'memory_gb': 0,
            'optimization_level': 'basic'
        }
        
        if platform.system() != 'Darwin':
            return system_info
        
        try:
            # Get CPU information
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True, timeout=5)
            cpu_info = result.stdout.strip()
            
            if 'Apple' in cpu_info:
                system_info['is_apple_silicon'] = True
                
                # Detect specific chip
                if 'M3' in cpu_info:
                    system_info['chip_type'] = 'M3'
                    system_info['optimization_level'] = 'maximum'
                    # M3 typically has 10 GPU cores (base model)
                    system_info['gpu_cores'] = 10
                elif 'M2' in cpu_info:
                    system_info['chip_type'] = 'M2'
                    system_info['optimization_level'] = 'high'
                    system_info['gpu_cores'] = 8
                elif 'M1' in cpu_info:
                    system_info['chip_type'] = 'M1'
                    system_info['optimization_level'] = 'high'
                    system_info['gpu_cores'] = 7
            
            # Get CPU core count
            result = subprocess.run(['sysctl', '-n', 'hw.ncpu'], 
                                  capture_output=True, text=True, timeout=5)
            system_info['cpu_cores'] = int(result.stdout.strip())
            
            # Get memory size
            result = subprocess.run(['sysctl', '-n', 'hw.memsize'], 
                                  capture_output=True, text=True, timeout=5)
            memory_bytes = int(result.stdout.strip())
            system_info['memory_gb'] = memory_bytes // (1024**3)
            
        except Exception as e:
            logger.debug(f"Erro ao detectar informações do sistema: {e}")
        
        return system_info
    
    @staticmethod
    def configure_mediapipe_for_m3() -> Tuple[Dict, Dict]:
        """
        Configure MediaPipe specifically for Mac M3 optimization
        Configurar MediaPipe especificamente para otimização do Mac M3
        """
        system_info = MacM3Optimizer.detect_mac_silicon()
        
        config = {
            'gpu_enabled': False,
            'metal_enabled': False,
            'performance_mode': 'cpu',
            'thread_count': 4,
            'recommendations': []
        }
        
        if system_info['is_apple_silicon']:
            # Enable GPU for Apple Silicon
            os.environ['MEDIAPIPE_DISABLE_GPU'] = '0'
            config['gpu_enabled'] = True
            config['metal_enabled'] = True
            config['performance_mode'] = 'gpu_accelerated'
            
            # M3-specific optimizations
            if system_info['chip_type'] == 'M3':
                # Enable all optimizations for M3
                os.environ['METAL_DEVICE_WRAPPER_TYPE'] = '1'
                os.environ['METAL_FORCE_INTEL_GPU'] = '0'
                os.environ['MEDIAPIPE_GPU_DEVICE'] = '0'
                
                # Set optimal thread count (M3 has 8 performance + 4 efficiency cores)
                optimal_threads = min(system_info['cpu_cores'], 8)
                config['thread_count'] = optimal_threads
                
                config['recommendations'].extend([
                    'GPU aceleração ativada (Metal Performance Shaders)',
                    f'Usando {optimal_threads} threads otimizadas',
                    f'{system_info["gpu_cores"]} cores de GPU disponíveis',
                    f'{system_info["memory_gb"]}GB RAM unificada'
                ])
            
            # General Apple Silicon optimizations
            os.environ['OMP_NUM_THREADS'] = str(config['thread_count'])
            os.environ['VECLIB_MAXIMUM_THREADS'] = str(config['thread_count'])
            
        else:
            # Fallback for non-Apple Silicon
            os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
            config['recommendations'].append('GPU não disponível - usando CPU otimizada')
        
        return config, system_info
    
    @staticmethod
    def setup_quiet_and_optimized():
        """
        Setup both quiet logging and GPU optimization
        Configurar logging silencioso e otimização de GPU
        """
        # Import logging utilities
        from .logging_config import enable_quiet_mode
        
        # Enable quiet mode first
        enable_quiet_mode()
        
        # Then configure GPU optimization
        config, system_info = MacM3Optimizer.configure_mediapipe_for_m3()
        
        return config, system_info


def print_optimization_report(config: Dict, system_info: Dict):
    """
    Print a detailed optimization report
    Imprimir relatório detalhado de otimização
    """
    print("\n🚀 RELATÓRIO DE OTIMIZAÇÃO DO SISTEMA")
    print("=" * 60)
    
    # System information
    print(f"💻 Sistema: {platform.system()} {platform.machine()}")
    if system_info['is_apple_silicon']:
        print(f"🔥 Chip: Apple {system_info['chip_type']}")
        print(f"🎮 GPU Cores: {system_info['gpu_cores']}")
        print(f"⚡ CPU Cores: {system_info['cpu_cores']}")
        print(f"💾 RAM: {system_info['memory_gb']}GB (Unificada)")
    
    # Configuration status
    print(f"\n⚙️ Configuração:")
    print(f"   GPU Ativada: {'✅ SIM' if config['gpu_enabled'] else '❌ NÃO'}")
    print(f"   Metal: {'✅ SIM' if config['metal_enabled'] else '❌ NÃO'}")
    print(f"   Modo: {config['performance_mode']}")
    print(f"   Threads: {config['thread_count']}")
    
    # Recommendations
    if config['recommendations']:
        print(f"\n📊 Otimizações Ativas:")
        for rec in config['recommendations']:
            print(f"   ✅ {rec}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test the optimization
    config, system_info = MacM3Optimizer.configure_mediapipe_for_m3()
    print_optimization_report(config, system_info)
