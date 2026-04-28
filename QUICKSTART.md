# QUICK START GUIDE
## AI Symptoms Analysis Using CNNs

### ⚡ 5-Minute Quick Start

#### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 2: Run Training
```bash
python train.py
```

This will:
- ✅ Load sample dataset
- ✅ Create CNN model
- ✅ Train for 10 epochs
- ✅ Save best model
- ✅ Display metrics

**Output:**
```
Test Accuracy: 0.7234
Test Precision: 0.7150
Test Recall: 0.7320
Model saved to: models/model_YYYYMMDD_HHMMSS/best_model.h5
```

#### Step 3: Make Predictions
```python
from inference import SymptomAnalyzer

analyzer = SymptomAnalyzer('models/best_model.h5')
result = analyzer.predict_single('path/to/image.jpg')

print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## 🚀 Next Steps

### 1. View Full Workflow
Open the Jupyter Notebook for comprehensive examples:
```bash
jupyter notebook notebooks/AI_Symptoms_Analysis_CNN.ipynb
```

### 2. Use Your Own Dataset
Replace the sample dataset:

```python
from data_preprocessing import SymptomDataLoader

loader = SymptomDataLoader(image_size=(224, 224))
X, y, classes = loader.load_images_from_directory('path/to/your/data')
```

### 3. Adjust Training Parameters
Edit `train.py` or use:

```python
from train import train_model

model, history, metrics = train_model(
    model_type='mobilenet',  # Lightweight model
    epochs=100,
    batch_size=16,
    learning_rate=0.0001
)
```

### 4. Use Pre-trained Models
```python
from cnn_model import create_mobilenet_model, compile_model

model = create_mobilenet_model(num_classes=5)
model = compile_model(model, learning_rate=0.001)
```

---

## 📊 Performance Expectations

### Dataset Size Impact
| Dataset Size | Expected Accuracy | Training Time |
|-------------|------------------|--------------|
| 100 samples | 60-70% | 1-2 min |
| 1,000 samples | 75-85% | 5-10 min |
| 10,000 samples | 85-95% | 30-60 min |
| 100,000+ samples | 95%+ | 2+ hours |

### Model Architecture Comparison
| Model | Speed | Accuracy | Memory |
|-------|-------|----------|--------|
| Simple CNN | 🔥🔥 Fast | ⭐⭐⭐ Good | 💾 Small |
| MobileNetV2 | 🔥🔥🔥 Fastest | ⭐⭐⭐⭐ Better | 💾 Smallest |
| ResNet50 | 🔥 Slower | ⭐⭐⭐⭐⭐ Best | 💾 Large |

---

## 🔍 Understanding the Output

### Training Output
```
Epoch 1/10
32/32 [==============================] - 2s
Train Loss: 0.6543 | Train Acc: 0.6800
Val Loss: 0.5234 | Val Acc: 0.7100
```

### Prediction Output
```python
{
    'predicted_class': 'Abnormal',
    'confidence': 0.8742,  # 87.42%
    'status': 'confident',
    'all_predictions': {
        'Normal': 0.1258,
        'Abnormal': 0.8742
    }
}
```

### Evaluation Metrics
- **Accuracy**: Overall correctness (TP + TN) / Total
- **Precision**: Correct positive predictions / All positive predictions
- **Recall**: Correct positive predictions / All actual positives
- **F1-Score**: Harmonic mean of Precision and Recall (0-1)
- **ROC-AUC**: Area under ROC curve (0-1, higher is better)

---

## ⚙️ Troubleshooting

### Problem: Slow Training
```python
# Solution 1: Use GPU
# Verify GPU: python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Solution 2: Reduce image size
image_size = (128, 128)  # Instead of (224, 224)

# Solution 3: Use lighter model
model_type='mobilenet'  # Instead of 'simple' or 'resnet'

# Solution 4: Increase batch size
batch_size=64  # Instead of 32
```

### Problem: Low Accuracy
```python
# Solution 1: Add more data
# Collect more training samples

# Solution 2: Increase training epochs
epochs=100  # Instead of 50

# Solution 3: Better data augmentation
# Edit AUGMENTATION_CONFIG in config.py

# Solution 4: Improve data quality
# Ensure images are properly labeled and clear

# Solution 5: Use transfer learning
model_type='resnet'  # Pre-trained model
```

### Problem: Out of Memory
```python
# Solution 1: Reduce batch size
batch_size=8  # Instead of 32

# Solution 2: Smaller image size
image_size = (128, 128)  # Instead of (224, 224)

# Solution 3: Use gradient checkpointing
# Built into some models

# Solution 4: Use lighter model
model_type='mobilenet'
```

---

## 📚 Learning Path

1. **Beginner** → Run `train.py` → Understand basic workflow
2. **Intermediate** → Use Jupyter notebook → Modify parameters
3. **Advanced** → Build custom models → Implement new features
4. **Expert** → Deploy → Production optimization

---

## 🎯 Common Tasks

### Task: Train with custom dataset
```python
from data_preprocessing import SymptomDataLoader
from train import train_model

loader = SymptomDataLoader()
X, y, classes = loader.load_images_from_directory('my_data')
X_train, X_val, X_test, y_train, y_val, y_test = loader.create_train_val_test_splits(X, y)

# Continue training...
```

### Task: Batch predictions
```python
from inference import SymptomAnalyzer

analyzer = SymptomAnalyzer('models/best_model.h5')
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = analyzer.predict_batch(images)

for result in results:
    print(f"{result['image_path']}: {result['predicted_class']}")
```

### Task: Get prediction explanation
```python
explanation = analyzer.explain_prediction('image.jpg')
print(explanation['top_predictions'])  # Top 3 predictions
print(explanation['recommendation'])   # Doctor recommendation
```

---

## 📞 Need Help?

1. **Read the full README.md** - Comprehensive documentation
2. **Check the Jupyter notebook** - Working examples
3. **Review config.py** - All configuration options
4. **Look at docstrings** - In-code documentation

---

**Happy Learning! 🚀**
