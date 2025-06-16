"""
Web interface modules for Photo Culling System
Interface web para sistema de classificação de imagens
"""

from .app import WebLabelingApp, create_app, start_web_server

__all__ = [
    'WebLabelingApp',
    'create_app', 
    'start_web_server'
]
