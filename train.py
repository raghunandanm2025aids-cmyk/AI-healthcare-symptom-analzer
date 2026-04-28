"""
Training script for CNN symptom analysis model
"""

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from cnn_model import create_simple_cnn, create_mobilenet_model, compile_model
from data_preprocessing import SymptomDataLoader, load_sample_dataset


def train_model(model_type='simple', epochs=50, batch_size=32, learning_rate=0.001):
    """
    Train CNN model for symptom analysis
    
    Args:
        model_type: 'simple', 'mobilenet', or 'resnet'
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    
    Returns:
        Trained model and training history
    """
    
    print("=" * 60)
    print("AI SYMPTOMS ANALYSIS - CNN MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading dataset...")
    loader = SymptomDataLoader(image_size=(224, 224), batch_size=batch_size)
    X, y = load_sample_dataset()
    print(f"✓ Loaded {len(X)} images with {len(np.unique(y))} classes")
    
    # Split data
    print("\n[2/4] Splitting dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = loader.create_train_val_test_splits(X, y)
    print(f"✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create datasets
    print("\n[3/4] Creating TensorFlow datasets...")
    train_dataset = loader.create_tf_dataset(X_train, y_train, augment=True, shuffle=True)
    val_dataset = loader.create_tf_dataset(X_val, y_val, augment=False, shuffle=False)
    test_dataset = loader.create_tf_dataset(X_test, y_test, augment=False, shuffle=False)
    print("✓ Datasets created successfully")
    
    # Create model
    print("\n[4/4] Creating model...")
    if model_type == 'mobilenet':
        model = create_mobilenet_model(input_shape=(224, 224, 3), num_classes=loader.num_classes)
        print(f"✓ Created MobileNetV2 transfer learning model")
    else:
        model = create_simple_cnn(input_shape=(224, 224, 3), num_classes=loader.num_classes)
        print(f"✓ Created simple CNN model")
    
    model = compile_model(model, learning_rate=learning_rate)
    
    # Model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Create callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models/model_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, 'logs'),
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    # Get class weights
    class_weights = loader.get_class_weights(y_train)
    print(f"\nClass weights: {class_weights}")
    
    # Train model
    print("\n" + "=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)
    
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_dataset)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    
    # Save final model
    final_model_path = os.path.join(model_dir, 'final_model.h5')
    model.save(final_model_path)
    print(f"\n✓ Model saved to {final_model_path}")
    
    # Plot training history
    plot_training_history(history, model_dir)
    
    return model, history, (test_loss, test_accuracy, test_precision, test_recall)


def plot_training_history(history, save_dir):
    """
    Plot and save training history graphs
    
    Args:
        history: Training history from model.fit()
        save_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history plot saved to {plot_path}")
    plt.close()


if __name__ == "__main__":
    # Train model with simple CNN architecture
    model, history, metrics = train_model(
        model_type='simple',
        epochs=10,  # Reduced for demo
        batch_size=32,
        learning_rate=0.001
    )
