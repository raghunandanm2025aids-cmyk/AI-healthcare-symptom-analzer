"""
Example usage script for AI Symptoms Analysis using CNNs

This script demonstrates various usage patterns:
1. Training a model
2. Making predictions
3. Evaluating performance
4. Batch processing
5. Saving and loading models
"""

import numpy as np
from pathlib import Path
from cnn_model import create_simple_cnn, compile_model
from data_preprocessing import SymptomDataLoader, load_sample_dataset
from inference import SymptomAnalyzer
import tensorflow as tf


def example_1_train_and_save():
    """Example 1: Train a model and save it"""
    print("\n" + "="*60)
    print("Example 1: Train and Save Model")
    print("="*60)
    
    # Load data
    loader = SymptomDataLoader(image_size=(224, 224), batch_size=32)
    X, y = load_sample_dataset()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = loader.create_train_val_test_splits(X, y)
    print(f"✓ Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Create and compile model
    model = create_simple_cnn(input_shape=(224, 224, 3), num_classes=2)
    model = compile_model(model, learning_rate=0.001)
    print("✓ Model created and compiled")
    
    # Create datasets
    train_dataset = loader.create_tf_dataset(X_train, y_train, augment=True)
    val_dataset = loader.create_tf_dataset(X_val, y_val, augment=False)
    
    # Train (short training for demo)
    print("✓ Starting training (5 epochs for demo)...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=5,
        verbose=0
    )
    
    # Save model
    model.save('models/example_model.h5')
    print("✓ Model saved to 'models/example_model.h5'")
    
    return model, X_test, y_test


def example_2_load_and_predict(X_test, y_test):
    """Example 2: Load a model and make predictions"""
    print("\n" + "="*60)
    print("Example 2: Load Model and Make Predictions")
    print("="*60)
    
    try:
        analyzer = SymptomAnalyzer(
            'models/example_model.h5',
            class_names=['Normal', 'Abnormal']
        )
        
        # Get a test image
        test_image = X_test[0]
        
        # Make prediction
        result = analyzer.predict_single(test_image)
        print(f"\nPrediction Result:")
        print(f"  Class: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Status: {result['status']}")
        print(f"  All probabilities: {result['all_predictions']}")
        
    except FileNotFoundError:
        print("⚠ Model file not found. Train a model first using example_1.")


def example_3_batch_prediction():
    """Example 3: Batch prediction on multiple images"""
    print("\n" + "="*60)
    print("Example 3: Batch Prediction")
    print("="*60)
    
    # Load sample data
    loader = SymptomDataLoader()
    X, y = load_sample_dataset(num_samples_per_class=10)
    
    try:
        analyzer = SymptomAnalyzer(
            'models/example_model.h5',
            class_names=['Normal', 'Abnormal']
        )
        
        print(f"✓ Making predictions on {len(X)} images...")
        
        # Predict on multiple images
        predictions = []
        for i, img in enumerate(X[:5]):  # First 5 images
            result = analyzer.predict_single(img)
            predictions.append(result)
            print(f"  Image {i+1}: {result['predicted_class']} ({result['confidence']:.2%})")
        
        # Calculate statistics
        confident = sum(1 for p in predictions if p['status'] == 'confident')
        print(f"\n✓ Confident predictions: {confident}/{len(predictions)}")
        
    except FileNotFoundError:
        print("⚠ Model file not found. Train a model first.")


def example_4_evaluate_model():
    """Example 4: Evaluate model performance"""
    print("\n" + "="*60)
    print("Example 4: Model Evaluation")
    print("="*60)
    
    # Load data
    loader = SymptomDataLoader(batch_size=32)
    X, y = load_sample_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.create_train_val_test_splits(X, y)
    
    try:
        # Load model
        model = tf.keras.models.load_model('models/example_model.h5')
        
        # Create test dataset
        test_dataset = loader.create_tf_dataset(X_test, y_test, augment=False)
        
        # Evaluate
        loss, accuracy, precision, recall = model.evaluate(test_dataset, verbose=0)
        
        print(f"\nTest Set Metrics:")
        print(f"  Loss:      {loss:.4f}")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        
    except FileNotFoundError:
        print("⚠ Model file not found. Train a model first.")


def example_5_custom_dataset():
    """Example 5: Work with custom dataset"""
    print("\n" + "="*60)
    print("Example 5: Custom Dataset Loading")
    print("="*60)
    
    loader = SymptomDataLoader(image_size=(224, 224))
    
    # Note: This example shows how to load from a directory
    # Your directory should have structure like:
    # data/
    #   ├── normal/
    #   │   ├── image1.jpg
    #   │   └── image2.jpg
    #   └── abnormal/
    #       ├── image3.jpg
    #       └── image4.jpg
    
    dataset_path = 'data'
    
    if Path(dataset_path).exists():
        print(f"✓ Loading from {dataset_path}...")
        # X, y, classes = loader.load_images_from_directory(dataset_path)
        # print(f"✓ Loaded {len(X)} images from {len(classes)} classes")
        print("  (Awaiting custom dataset)")
    else:
        print(f"ℹ Directory '{dataset_path}' not found.")
        print(f"  Create the following structure to use custom datasets:")
        print(f"  {dataset_path}/")
        print(f"  ├── class1/")
        print(f"  │   ├── image1.jpg")
        print(f"  │   └── image2.jpg")
        print(f"  └── class2/")
        print(f"      ├── image3.jpg")
        print(f"      └── image4.jpg")


def example_6_hyperparameter_tuning():
    """Example 6: Experiment with different hyperparameters"""
    print("\n" + "="*60)
    print("Example 6: Hyperparameter Tuning")
    print("="*60)
    
    print("\nCommon hyperparameter adjustments:\n")
    
    configs = [
        {"name": "Fast Training", "epochs": 10, "batch_size": 64, "lr": 0.01},
        {"name": "Balanced", "epochs": 50, "batch_size": 32, "lr": 0.001},
        {"name": "Careful Training", "epochs": 100, "batch_size": 16, "lr": 0.0001},
    ]
    
    for config in configs:
        print(f"\n{config['name']}:")
        print(f"  Epochs: {config['epochs']}")
        print(f"  Batch Size: {config['batch_size']}")
        print(f"  Learning Rate: {config['lr']}")
        print(f"  Estimated Training Time: ~{config['epochs'] * 2 // 10 + 1} minutes")


def main():
    """Run all examples"""
    print("\n" + "🚀 "*30)
    print("AI SYMPTOMS ANALYSIS - USAGE EXAMPLES")
    print("🚀 "*30)
    
    # Create models directory if it doesn't exist
    Path('models').mkdir(exist_ok=True)
    
    # Run examples
    try:
        # Example 1: Train and save
        model, X_test, y_test = example_1_train_and_save()
        
        # Example 2: Load and predict
        example_2_load_and_predict(X_test, y_test)
        
        # Example 3: Batch prediction
        example_3_batch_prediction()
        
        # Example 4: Evaluate
        example_4_evaluate_model()
        
        # Example 5: Custom dataset
        example_5_custom_dataset()
        
        # Example 6: Hyperparameter tuning
        example_6_hyperparameter_tuning()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("✓ All examples completed!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Open QUICKSTART.md for quick reference")
    print("  2. Check README.md for detailed documentation")
    print("  3. Run Jupyter notebook for interactive learning")
    print("  4. Modify this script for your specific use case")
    print("\n")


if __name__ == "__main__":
    main()
