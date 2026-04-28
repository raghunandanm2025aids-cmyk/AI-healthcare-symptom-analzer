# AI Symptoms Analysis Package
"""
AI Symptoms Analysis using Convolutional Neural Networks (CNNs)

This package provides a complete workflow for analyzing medical symptoms
from images using deep learning. It includes data preprocessing, model
training, evaluation, and inference capabilities.

Main modules:
- cnn_model: CNN model architectures
- data_preprocessing: Data loading and preprocessing
- train: Training script
- inference: Prediction and inference utilities
- config: Configuration settings

Example:
    Basic training workflow:
    
    >>> from data_preprocessing import SymptomDataLoader
    >>> from train import train_model
    >>> 
    >>> # Train model
    >>> model, history, metrics = train_model(
    ...     model_type='simple',
    ...     epochs=50,
    ...     batch_size=32
    ... )
    
    Making predictions:
    
    >>> from inference import SymptomAnalyzer
    >>> 
    >>> analyzer = SymptomAnalyzer('models/best_model.h5')
    >>> result = analyzer.predict_single('image.jpg')
    >>> print(result['predicted_class'], result['confidence'])
"""

__version__ = '1.0.0'
__author__ = 'AI Learning Project'
__license__ = 'Educational Use'

from cnn_model import (
    create_simple_cnn,
    create_mobilenet_model,
    create_resnet_model,
    compile_model
)

from data_preprocessing import SymptomDataLoader

from inference import SymptomAnalyzer

__all__ = [
    'create_simple_cnn',
    'create_mobilenet_model',
    'create_resnet_model',
    'compile_model',
    'SymptomDataLoader',
    'SymptomAnalyzer',
]
