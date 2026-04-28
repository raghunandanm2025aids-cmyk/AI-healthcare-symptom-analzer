"""
Configuration file for AI Symptoms Analysis project
"""

# Model Configuration
MODEL_CONFIG = {
    'input_shape': (224, 224, 3),
    'num_classes': 2,
    'architecture': 'simple_cnn',  # Options: 'simple_cnn', 'mobilenet', 'resnet'
}

# Training Configuration
TRAINING_CONFIG = {
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'validation_split': 0.15,
    'test_split': 0.15,
}

# Data Configuration
DATA_CONFIG = {
    'image_size': (224, 224),
    'normalize': True,
    'normalization_range': (0, 1),
    'augmentation': True,
}

# Augmentation Configuration
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': 0.2,
    'shear_range': 0.15,
    'fill_mode': 'nearest',
}

# Class Configuration
CLASS_NAMES = {
    2: ['Normal', 'Abnormal'],
    3: ['Healthy', 'Mild', 'Severe'],
    4: ['Class_0', 'Class_1', 'Class_2', 'Class_3'],
}

# Callback Configuration
CALLBACK_CONFIG = {
    'early_stopping': {
        'monitor': 'val_loss',
        'patience': 10,
        'restore_best_weights': True,
    },
    'reduce_lr': {
        'monitor': 'val_loss',
        'factor': 0.5,
        'patience': 5,
        'min_lr': 0.00001,
    },
}

# Inference Configuration
INFERENCE_CONFIG = {
    'confidence_threshold': 0.5,  # Minimum confidence for prediction
    'top_k': 3,  # Number of top predictions to return
    'use_best_model': True,  # Use best model if available
}

# Paths
PATHS = {
    'data_dir': 'data',
    'model_dir': 'models',
    'notebook_dir': 'notebooks',
    'log_dir': 'logs',
}

# Default class names
DEFAULT_CLASS_NAMES = ['Normal', 'Abnormal']

# Random seed for reproducibility
RANDOM_SEED = 42
