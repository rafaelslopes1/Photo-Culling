#!/usr/bin/env python3
"""
Logging Configuration Utilities
Configuração de logging para o sistema

Provides clean logging setup and noise reduction
"""

import os
import sys
import logging
import warnings
from contextlib import contextmanager
import platform
import subprocess


class CleanLogger:
    """
    Clean logging setup with MediaPipe noise reduction
    Sistema de logging limpo com redução de ruído do MediaPipe
    """
    
    @staticmethod
    def setup_clean_logging(level=logging.INFO, 
                           suppress_tensorflow=True,
                           suppress_mediapipe=True):
        """
        Configure clean logging with minimal noise
        
        Args:
            level: Logging level
            suppress_tensorflow: Suppress TensorFlow messages
            suppress_mediapipe: Suppress MediaPipe internal messages
        """
        
        # Basic logging setup
        logging.basicConfig(
            level=level,
            format='%(levelname)s:%(name)s:%(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        
        if suppress_tensorflow:
            # Suppress TensorFlow INFO and WARNING messages
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            logging.getLogger('tensorflow').setLevel(logging.ERROR)
        
        if suppress_mediapipe:
            # Suppress MediaPipe internal warnings
            warnings.filterwarnings('ignore', category=UserWarning, module='mediapipe')
            
            # Suppress ABSL logging (used by MediaPipe)
            try:
                import absl.logging
                absl.logging.set_verbosity(absl.logging.ERROR)
            except ImportError:
                pass
    
    @staticmethod
    @contextmanager 
    def quiet_mode():
        """
        Context manager for completely quiet execution
        Gerenciador de contexto para execução silenciosa
        """
        original_stderr = sys.stderr
        original_stdout = sys.stdout
        
        try:
            # Redirect stderr to devnull during MediaPipe initialization
            with open(os.devnull, 'w') as devnull:
                sys.stderr = devnull
                yield
        finally:
            sys.stderr = original_stderr
            sys.stdout = original_stdout
    
    @staticmethod
    def setup_production_logging():
        """
        Setup logging for production environment
        Configuração de logging para ambiente de produção
        """
        CleanLogger.setup_clean_logging(
            level=logging.WARNING,
            suppress_tensorflow=True,
            suppress_mediapipe=True
        )
        
        # Additional production settings
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)


def setup_quiet_mediapipe():
    """
    Setup MediaPipe with minimal logging output
    Configurar MediaPipe com saída mínima de logs
    """
    # Environment variables to reduce MediaPipe verbosity
    os.environ.setdefault('GLOG_minloglevel', '3')  # Suppress all GLOG messages
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # Suppress all TF messages
    os.environ.setdefault('GLOG_logtostderr', '0')  # Don't log to stderr
    
    # MediaPipe specific environment variables
    os.environ.setdefault('MEDIAPIPE_DISABLE_GPU', '0')  # Keep GPU enabled
    os.environ.setdefault('ABSL_STDERRTHRESHOLD', '3')  # Suppress ABSL logs
    
    # Suppress specific warnings with more comprehensive patterns
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    # Suppress specific MediaPipe messages
    import logging
    logging.getLogger('absl').setLevel(logging.CRITICAL)
    logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
    logging.getLogger('mediapipe').setLevel(logging.WARNING)


# Easy import function
def enable_quiet_mode():
    """
    Enable quiet mode with one function call
    Ativar modo silencioso com uma chamada de função
    """
    CleanLogger.setup_clean_logging(
        level=logging.INFO,
        suppress_tensorflow=True,
        suppress_mediapipe=True
    )
    setup_quiet_mediapipe()


def setup_optimal_performance():
    """
    Setup optimal performance configuration for the current system
    Configurar performance otimizada para o sistema atual
    """
    # Setup quiet logging
    enable_quiet_mode()
    
    # Configure GPU optimization using dedicated module
    try:
        from .gpu_optimizer import MacM3Optimizer
        gpu_config, gpu_info = MacM3Optimizer.configure_mediapipe_for_m3()
        return gpu_config, gpu_info
    except ImportError:
        # Fallback if gpu_optimizer not available
        return {'gpu_enabled': False}, {'is_apple_silicon': False}


if __name__ == "__main__":
    # Test the quiet mode
    print("🧪 Testando modo de logging limpo...")
    enable_quiet_mode()
    print("✅ Modo silencioso ativado!")
