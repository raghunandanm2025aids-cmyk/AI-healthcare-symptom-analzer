"""
Inference module for making predictions with trained models
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json


class SymptomAnalyzer:
    """Perform inference with trained CNN model"""
    
    def __init__(self, model_path, class_names=None):
        """
        Initialize analyzer with trained model
        
        Args:
            model_path: Path to saved model (.h5 or SavedModel format)
            class_names: List of class names (e.g., ['Normal', 'Abnormal'])
        """
        print(f"Loading model from {model_path}...")
        self.model = keras.models.load_model(model_path)
        self.class_names = class_names or ['Class_0', 'Class_1']
        self.image_size = (224, 224)
        print("✓ Model loaded successfully")
    
    def preprocess_image(self, image_path):
        """
        Load and preprocess image for prediction
        
        Args:
            image_path: Path to image file
        
        Returns:
            Preprocessed image array (4D: batch size 1)
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, self.image_size)
        
        # Normalize
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict_single(self, image_path, confidence_threshold=0.5):
        """
        Predict symptom/disease for a single image
        
        Args:
            image_path: Path to image
            confidence_threshold: Minimum confidence for prediction
        
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        img = self.preprocess_image(image_path)
        
        # Predict
        predictions = self.model.predict(img, verbose=0)
        
        # Get results
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        # Check confidence threshold
        if confidence < confidence_threshold:
            return {
                'status': 'uncertain',
                'message': f'Confidence {confidence:.2%} below threshold {confidence_threshold:.0%}',
                'predicted_class': self.class_names[class_idx],
                'confidence': confidence,
                'all_predictions': {
                    self.class_names[i]: float(pred) 
                    for i, pred in enumerate(predictions[0])
                }
            }
        
        return {
            'status': 'success',
            'predicted_class': self.class_names[class_idx],
            'confidence': confidence,
            'all_predictions': {
                self.class_names[i]: float(pred) 
                for i, pred in enumerate(predictions[0])
            }
        }
    
    def predict_batch(self, image_paths, confidence_threshold=0.5):
        """
        Predict symptoms for multiple images
        
        Args:
            image_paths: List of image paths
            confidence_threshold: Minimum confidence for prediction
        
        Returns:
            List of prediction results
        """
        results = []
        
        for idx, image_path in enumerate(image_paths):
            print(f"Processing {idx+1}/{len(image_paths)}: {image_path}")
            try:
                result = self.predict_single(image_path, confidence_threshold)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                results.append({
                    'status': 'error',
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def predict_from_array(self, image_array):
        """
        Predict from numpy array (useful for live camera feed)
        
        Args:
            image_array: Numpy array of shape (height, width, 3)
        
        Returns:
            Prediction result
        """
        # Resize
        img = cv2.resize(image_array, self.image_size)
        
        # Normalize
        img = img.astype('float32') / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Predict
        predictions = self.model.predict(img, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        return {
            'predicted_class': self.class_names[class_idx],
            'confidence': confidence,
            'all_predictions': {
                self.class_names[i]: float(pred) 
                for i, pred in enumerate(predictions[0])
            }
        }
    
    def explain_prediction(self, image_path):
        """
        Provide detailed explanation of prediction
        
        Args:
            image_path: Path to image
        
        Returns:
            Detailed prediction information
        """
        img = self.preprocess_image(image_path)
        predictions = self.model.predict(img, verbose=0)
        
        # Sort by confidence
        sorted_idx = np.argsort(predictions[0])[::-1]
        
        explanation = {
            'image_path': image_path,
            'top_predictions': [
                {
                    'class': self.class_names[idx],
                    'confidence': float(predictions[0][idx]),
                    'percentage': f"{predictions[0][idx]*100:.2f}%"
                }
                for idx in sorted_idx[:min(3, len(self.class_names))]
            ],
            'recommendation': self._get_recommendation(sorted_idx[0], predictions[0][sorted_idx[0]])
        }
        
        return explanation
    
    def _get_recommendation(self, predicted_class_idx, confidence):
        """Generate recommendation based on prediction"""
        confidence_pct = confidence * 100
        
        if confidence_pct >= 90:
            return "High confidence prediction. Likely diagnosis: " + self.class_names[predicted_class_idx]
        elif confidence_pct >= 70:
            return "Moderate confidence. Further investigation recommended."
        else:
            return "Low confidence. Additional images or medical consultation advised."


def demo_inference():
    """Demo function showing inference usage"""
    
    print("\n" + "=" * 60)
    print("SYMPTOM ANALYSIS - INFERENCE DEMO")
    print("=" * 60)
    
    # This is a demo - in real usage, provide actual model and images
    print("\nTo use this module:")
    print("1. Train a model using train.py")
    print("2. Initialize analyzer: analyzer = SymptomAnalyzer('path/to/model.h5', class_names=['Normal', 'Abnormal'])")
    print("3. Make predictions:")
    print("   - Single image: result = analyzer.predict_single('image.jpg')")
    print("   - Batch: results = analyzer.predict_batch(['img1.jpg', 'img2.jpg'])")
    print("   - Get explanation: analyzer.explain_prediction('image.jpg')")
    

if __name__ == "__main__":
    demo_inference()
