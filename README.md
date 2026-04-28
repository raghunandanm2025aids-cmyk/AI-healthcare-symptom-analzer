# AI Symptoms Analysis Using Convolutional Neural Networks (CNNs)

A complete deep learning project for analyzing medical symptoms from images using Convolutional Neural Networks. This project includes data preprocessing, model training, evaluation, and inference capabilities.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Features](#features)
- [Results](#results)
- [Future Enhancements](#future-enhancements)

## 🎯 Overview

This project demonstrates:
- **Image Classification**: CNN-based medical image classification
- **Deep Learning**: Building and training neural networks from scratch
- **Transfer Learning**: Options for using pre-trained models (MobileNetV2, ResNet50)
- **Model Evaluation**: Comprehensive metrics (accuracy, precision, recall, F1-score, ROC-AUC)
- **Production Ready**: Inference pipeline with confidence thresholds
- **Data Augmentation**: Techniques to improve model robustness
- **Visualization**: Training history, confusion matrices, and prediction visualizations

**Use Cases:**
- Medical imaging analysis
- Disease detection from symptoms
- Skin disease classification
- Chest X-ray analysis
- General symptom-based diagnostics

## 📁 Project Structure

```
python project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── cnn_model.py                # CNN model definitions
├── data_preprocessing.py        # Data loading and preprocessing
├── train.py                     # Training script
├── inference.py                 # Inference and prediction module
│
├── notebooks/
│   └── AI_Symptoms_Analysis_CNN.ipynb    # Jupyter notebook with full workflow
│
├── data/                        # Data directory
│   ├── raw/                     # Raw images organized by class
│   │   ├── normal/
│   │   └── abnormal/
│   └── processed/               # Processed data (if needed)
│
└── models/                      # Saved models
    └── model_YYYYMMDD_HHMMSS/
        ├── best_model.h5
        ├── final_model.h5
        ├── training_history.png
        └── logs/
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- CUDA 11.8+ (optional, for GPU acceleration)

### Step 1: Clone/Download the Project

```bash
cd "c:\Users\Raghunandan\OneDrive\Desktop\SECE FILE\python project"
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## 💻 Usage

### Option 1: Using Jupyter Notebook (Recommended for Learning)

```bash
jupyter notebook notebooks/AI_Symptoms_Analysis_CNN.ipynb
```

This notebook includes:
- Complete data loading and exploration
- Model architecture visualization
- Training with callbacks and monitoring
- Comprehensive evaluation metrics
- Prediction visualization

### Option 2: Using Training Script

```bash
python train.py
```

This will:
1. Load the sample dataset
2. Create and compile the model
3. Train for 10 epochs (adjustable)
4. Save the best model
5. Generate training history plots
6. Evaluate on test set

### Option 3: Using Inference Module

```python
from inference import SymptomAnalyzer

# Initialize analyzer
analyzer = SymptomAnalyzer(
    model_path='models/best_model.h5',
    class_names=['Normal', 'Abnormal']
)

# Single prediction
result = analyzer.predict_single('path/to/image.jpg')
print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch prediction
results = analyzer.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])

# Get detailed explanation
explanation = analyzer.explain_prediction('path/to/image.jpg')
print(explanation)
```

## 🧠 Model Architecture

### Simple CNN (Default)

```
Input (224×224×3)
    ↓
Conv2D(32, 3×3) + BatchNorm + ReLU
Conv2D(32, 3×3) + BatchNorm + ReLU
MaxPooling2D(2×2) + Dropout(0.25)
    ↓
Conv2D(64, 3×3) + BatchNorm + ReLU
Conv2D(64, 3×3) + BatchNorm + ReLU
MaxPooling2D(2×2) + Dropout(0.25)
    ↓
Conv2D(128, 3×3) + BatchNorm + ReLU
Conv2D(128, 3×3) + BatchNorm + ReLU
MaxPooling2D(2×2) + Dropout(0.25)
    ↓
Conv2D(256, 3×3) + BatchNorm + ReLU
MaxPooling2D(2×2) + Dropout(0.25)
    ↓
GlobalAveragePooling2D()
    ↓
Dense(512) + BatchNorm + Dropout(0.5)
Dense(256) + BatchNorm + Dropout(0.5)
Dense(128) + BatchNorm + Dropout(0.3)
    ↓
Dense(num_classes, softmax) → Output
```

**Parameters:** ~15-20 Million

### Available Pre-trained Models

1. **Simple CNN** (Default)
   - ~20M parameters
   - Fastest training
   - Good for small datasets

2. **MobileNetV2** (Transfer Learning)
   - Lightweight (~3.5M parameters)
   - Fast inference
   - Best for mobile/edge deployment

3. **ResNet50** (Transfer Learning)
   - Powerful (~24M parameters)
   - Best accuracy
   - Requires more data

```python
from cnn_model import create_mobilenet_model, create_resnet_model

# Use MobileNetV2
model = create_mobilenet_model(num_classes=2)

# Use ResNet50
model = create_resnet_model(num_classes=2)
```

## ✨ Features

### Data Preprocessing
- ✅ Automatic image resizing to (224×224)
- ✅ Pixel normalization to [0, 1]
- ✅ Train/Val/Test splitting (70/15/15)
- ✅ Data augmentation (rotation, flip, zoom, translation)
- ✅ Automatic class weight calculation for imbalanced datasets

### Model Training
- ✅ EarlyStopping to prevent overfitting
- ✅ ReduceLROnPlateau for adaptive learning rate
- ✅ ModelCheckpoint to save best model
- ✅ TensorBoard integration for visualization
- ✅ Batch normalization for faster convergence
- ✅ Dropout regularization

### Evaluation
- ✅ Accuracy, Precision, Recall metrics
- ✅ F1-Score calculation
- ✅ ROC-AUC score
- ✅ Confusion Matrix visualization
- ✅ Classification Report (per-class metrics)

### Inference
- ✅ Single image prediction
- ✅ Batch prediction
- ✅ Confidence thresholds
- ✅ Detailed explanation output
- ✅ Real-time camera feed support

### Visualization
- ✅ Sample image display
- ✅ Training history plots (accuracy, loss, precision, recall)
- ✅ Confusion matrix heatmap
- ✅ Prediction visualization with confidence scores

## 📊 Results

### Expected Performance (on sample dataset)

```
Test Accuracy:  ~75-85% (varies with dataset)
Test Precision: ~0.75-0.85
Test Recall:    ~0.75-0.85
Test F1-Score:  ~0.75-0.85
ROC-AUC:        ~0.80-0.90
```

**Note:** Actual performance depends on:
- Dataset size and quality
- Class balance
- Image resolution
- Number of training epochs
- Model architecture choice

## 🔧 Configuration

### Adjustable Parameters

Edit parameters in `train.py`:

```python
# Training parameters
epochs = 50              # Number of training epochs
batch_size = 32          # Batch size for training
learning_rate = 0.001    # Learning rate for optimizer

# Model parameters
image_size = (224, 224)  # Input image size
num_classes = 2          # Number of output classes

# Data split
train_size = 0.70        # 70% training
val_size = 0.15          # 15% validation
test_size = 0.15         # 15% testing
```

### Data Augmentation

Edit in `data_preprocessing.py`:

```python
# Augmentation parameters
rotation_range = 20
width_shift_range = 0.2
height_shift_range = 0.2
horizontal_flip = True
zoom_range = 0.2
shear_range = 0.15
```

## 📈 Loading Real Medical Datasets

### Option 1: Using Medical Image Datasets

```python
from data_preprocessing import SymptomDataLoader

loader = SymptomDataLoader(image_size=(224, 224), batch_size=32)

# Load from directory structure:
# dataset/
#   ├── normal/
#   │   ├── img1.jpg
#   │   └── img2.jpg
#   └── abnormal/
#       ├── img3.jpg
#       └── img4.jpg

X, y, classes = loader.load_images_from_directory('path/to/dataset')
```

### Option 2: Public Medical Datasets

Popular datasets for medical image analysis:

1. **ImageNet Medical Subset**
   - https://www.image-net.org/

2. **ChexPert** (Chest X-rays)
   - https://stanfordmlgroup.github.io/competitions/chexpert/

3. **ISIC** (Skin Lesions)
   - https://www.isic-archive.com/

4. **CIFAR-10/100**
   - https://www.cs.toronto.edu/~kriz/cifar.html

5. **Stanford COVID-19 X-ray Dataset**
   - https://github.com/ieee8023/covid-chexpert

## 🚨 Troubleshooting

### Issue: Out of Memory (OOM) Error
```python
# Reduce batch size
batch_size = 16  # or lower

# Reduce image size
image_size = (128, 128)

# Use MobileNetV2 instead of ResNet50
```

### Issue: Model Not Converging
```python
# Increase training epochs
epochs = 100

# Reduce learning rate
learning_rate = 0.0001

# Add more data augmentation
# Increase dropout rates
```

### Issue: GPU Not Detected
```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If not detected, install GPU support:
pip install tensorflow[and-cuda]
```

## 🔮 Future Enhancements

- [ ] 3D CNN for volumetric medical images
- [ ] Attention mechanisms (Self-Attention, Channel Attention)
- [ ] Model explainability (Grad-CAM, LIME, SHAP)
- [ ] Multi-task learning (simultaneous symptoms)
- [ ] Federated learning for privacy
- [ ] Real-time video stream analysis
- [ ] Web API deployment (Flask/FastAPI)
- [ ] Mobile app integration
- [ ] Automated hyperparameter tuning (Hyperopt, Ray Tune)
- [ ] Ensemble methods (voting, stacking)
- [ ] Active learning for continuous improvement
- [ ] Confidence calibration

## 📚 Learning Resources

### CNN Fundamentals
- [Stanford CS231n](http://cs231n.stanford.edu/)
- [Fast.ai Deep Learning Course](https://course.fast.ai/)
- [TensorFlow Documentation](https://www.tensorflow.org/learn)

### Medical Image Analysis
- [Stanford Lecture on Medical Imaging](https://www.youtube.com/watch?v=FODpRvHuFiA)
- [Deep Learning for Medical Image Analysis](https://arxiv.org/abs/2102.04747)

### Implementation Guides
- [Keras Documentation](https://keras.io/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Kaggle Competitions](https://www.kaggle.com/)

## 📝 License

This project is provided for educational purposes.

## 👨‍💻 Authors

Created as a comprehensive AI/ML learning project demonstration.

## ✉️ Support

For issues or questions:
1. Check the troubleshooting section
2. Review the Jupyter notebook for examples
3. Consult the documentation in code files

---

**Happy Learning! 🚀**

This project provides a solid foundation for medical symptom analysis using CNNs and can be extended for various real-world applications.
