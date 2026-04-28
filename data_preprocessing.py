"""
Data Preprocessing and Loading Module
Handles image loading, augmentation, and dataset preparation
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
from pathlib import Path


class SymptomDataLoader:
    """Handles loading and preprocessing symptom images"""
    
    def __init__(self, image_size=(224, 224), batch_size=32):
        """
        Initialize data loader
        
        Args:
            image_size: Target image size (height, width)
            batch_size: Batch size for training
        """
        self.image_size = image_size
        self.batch_size = batch_size
        self.class_names = None
        self.num_classes = None
    
    def load_images_from_directory(self, directory, classes=None):
        """
        Load images from directory structure: directory/class_name/image.jpg
        
        Args:
            directory: Path to directory containing class subdirectories
            classes: List of class names (if None, all subdirectories are used)
        
        Returns:
            images, labels, class_names arrays
        """
        images = []
        labels = []
        
        if classes is None:
            classes = sorted([d for d in os.listdir(directory) 
                            if os.path.isdir(os.path.join(directory, d))])
        
        self.class_names = classes
        self.num_classes = len(classes)
        
        for label, class_name in enumerate(classes):
            class_dir = os.path.join(directory, class_name)
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    img_path = os.path.join(class_dir, img_name)
                    try:
                        # Load and resize image
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, self.image_size)
                            images.append(img)
                            labels.append(label)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
        
        return np.array(images), np.array(labels), self.class_names
    
    def create_train_val_test_splits(self, X, y, train_size=0.7, val_size=0.15):
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Images array
            y: Labels array
            train_size: Proportion for training (0-1)
            val_size: Proportion for validation (0-1)
        
        Returns:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        test_size = 1 - train_size - val_size
        
        # First split: train + val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train vs val
        val_ratio = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_augmentation(self):
        """
        Create data augmentation generator for training
        
        Returns:
            ImageDataGenerator for augmentation
        """
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.15,
            fill_mode='nearest'
        )
    
    def create_tf_dataset(self, images, labels, augment=False, shuffle=True):
        """
        Create TensorFlow dataset from images and labels
        
        Args:
            images: Images array
            labels: Labels array
            augment: Whether to apply data augmentation
            shuffle: Whether to shuffle the dataset
        
        Returns:
            TensorFlow Dataset
        """
        # Convert labels to one-hot encoding
        labels_onehot = tf.keras.utils.to_categorical(labels, self.num_classes)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((images, labels_onehot))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(images))
        
        if augment:
            augmentation_layer = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
                tf.keras.layers.RandomTranslation(0.1, 0.1),
            ])
            dataset = dataset.map(lambda x, y: (augmentation_layer(x), y))
        
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_class_weights(self, y):
        """
        Calculate class weights to handle imbalanced datasets
        
        Args:
            y: Labels array
        
        Returns:
            Dictionary of class weights
        """
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        class_weights = {}
        for class_idx, count in zip(unique, counts):
            class_weights[class_idx] = total / (len(unique) * count)
        
        return class_weights


def load_sample_dataset():
    """
    Create a sample dataset for demonstration (Chest X-Ray like structure)
    
    Returns:
        Sample images and labels
    """
    # Generate synthetic data for demonstration
    num_samples_per_class = 50
    image_size = (224, 224, 3)
    num_classes = 2  # Normal, Abnormal
    
    X = []
    y = []
    
    for class_id in range(num_classes):
        for _ in range(num_samples_per_class):
            # Generate synthetic image
            img = np.random.randint(0, 256, size=image_size, dtype=np.uint8)
            X.append(img)
            y.append(class_id)
    
    return np.array(X), np.array(y)


if __name__ == "__main__":
    # Example usage
    loader = SymptomDataLoader(image_size=(224, 224), batch_size=32)
    
    # Create sample data
    X, y = load_sample_dataset()
    print(f"Loaded {len(X)} images with {len(np.unique(y))} classes")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = loader.create_train_val_test_splits(X, y)
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create datasets
    train_dataset = loader.create_tf_dataset(X_train, y_train, augment=True, shuffle=True)
    val_dataset = loader.create_tf_dataset(X_val, y_val, augment=False, shuffle=False)
    
    print(f"Train dataset batches: {len(train_dataset)}")
    print(f"Val dataset batches: {len(val_dataset)}")
