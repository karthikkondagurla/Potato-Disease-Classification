# Potato Disease Classification

A deep learning project for potato disease classification using Convolutional Neural Networks (CNN). This project classifies potato leaf images into three categories: **Early Blight**, **Late Blight**, and **Healthy**.

## Overview

Potato diseases, particularly Early Blight and Late Blight, can cause significant crop losses if not identified and managed early. This project uses a trained CNN model to automatically classify potato leaf images, enabling farmers to detect diseases quickly and take preventive measures.

### Dataset

- **Source**: PlantVillage Dataset (publicly available)
- **Total Images**: 2,152
- **Classes**: 3
  - Potato Early Blight
  - Potato Late Blight
  - Potato Healthy
- **Image Size**: 256x256 pixels
- **Train/Validation/Test Split**: 80/10/10

## Model Architecture

The project implements a Convolutional Neural Network (CNN) with the following architecture:

### Layers:
1. **Input Layer**: Resizing (256×256) + Rescaling (0-1 normalization)
2. **Data Augmentation**: Random Flip (horizontal and vertical) + Random Rotation (20%)
3. **Convolutional Blocks**:
   - Conv2D (32 filters, 3×3 kernel) → MaxPooling (2×2)
   - Conv2D (64 filters, 3×3 kernel) → MaxPooling (2×2)
   - Conv2D (64 filters, 3×3 kernel) → MaxPooling (2×2)
4. **Flattening Layer**
5. **Dense Layers**: 64 neurons (ReLU) → 3 neurons (Softmax)

### Total Parameters: 3.7M

## Performance Metrics

- **Training Accuracy**: 92.51%
- **Validation Accuracy**: ~96%
- **Test Accuracy**: 89.66%
- **Training Loss**: Minimized over 50 epochs

## Project Workflow

### 1. Data Collection & Preprocessing
- Downloaded PlantVillage dataset from GitHub
- Filtered and organized potato disease images
- Loaded images into TensorFlow dataset
- Normalized image sizes to 256×256
- Applied pixel normalization (0-1 range)

### 2. Data Exploration
- Visualized sample images from each class
- Verified dataset balance and distribution
- Explored image characteristics

### 3. Data Augmentation
- **Random Flip**: Horizontal and vertical flips
- **Random Rotation**: Up to 20% rotation
- Benefits: Prevents overfitting and improves model generalization

### 4. Model Building
- Built CNN architecture with sequential layers
- Compiled with Adam optimizer
- Used SparseCategoricalCrossentropy loss function
- Optimized for multi-class classification

### 5. Model Training
- Trained for 50 epochs
- Batch size: 32
- Used training and validation datasets
- Monitored accuracy and loss metrics

### 6. Model Evaluation
- Evaluated on test dataset
- Generated predictions on sample images
- Visualized training history (accuracy and loss curves)

### 7. Model Saving
- Saved trained model in `.keras` format
- Implemented version control for models
- Location: `saved_models/potato_model_v1.keras`

## Technologies Used

- **Deep Learning Framework**: TensorFlow/Keras
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib
- **Environment**: Google Colab (with GPU acceleration)
- **Language**: Python 3.12

## Dependencies

```
tensorflow==2.19.0
numpy
matplotlib
pillow
```

## Installation

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/karthikkondagurla/Potato-Disease-Classification.git
cd Potato-Disease-Classification
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

The complete training notebook is provided. To train the model:

1. Open `Potato_Disease_Classification_Complete.ipynb`
2. Follow the cells in order
3. The model will be saved to `saved_models/` directory

### Making Predictions

To make predictions on new potato leaf images:

```python
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('saved_models/potato_model_v1.keras')

# Load and preprocess image
img = Image.open('path/to/potato_leaf.jpg').convert('RGB')
img = img.resize((256, 256))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, 0)

# Make prediction
prediction = model.predict(img_array)
class_names = ['Early Blight', 'Healthy', 'Late Blight']
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
```

## Project Files

- `Potato_Disease_Classification_Complete.ipynb` - Complete notebook with all steps
- `saved_models/` - Directory containing trained model
- `README.md` - This file
- `requirements.txt` - Python dependencies

## Key Features

✓ Complete end-to-end deep learning pipeline
✓ Data augmentation for improved generalization
✓ Performance visualization with training curves
✓ High test accuracy (89.66%)
✓ Well-documented code with explanations
✓ Model versioning system
✓ Easy-to-use prediction interface

## Results & Analysis

### Model Performance
The trained model achieves excellent accuracy on all datasets:
- Consistently high validation accuracy (~96%) indicates good generalization
- Test accuracy of 89.66% shows reliable performance on unseen data
- Minimal overfitting observed in training curves

### Class Distribution
The dataset contains three classes:
1. **Early Blight**: Fungal disease causing spotted leaf lesions
2. **Late Blight**: More aggressive fungal disease
3. **Healthy**: Normal, disease-free potato leaves

## Future Improvements

- [ ] Deploy model as a web application
- [ ] Create a mobile app for farmers
- [ ] Implement transfer learning with pre-trained models (ResNet, VGG)
- [ ] Add more potato disease classes
- [ ] Expand dataset with more images
- [ ] Implement explainability features (visualization of important regions)
- [ ] Add real-time camera input for prediction
- [ ] Deploy on edge devices for offline use

## Challenges & Solutions

### Challenge 1: Dataset Download
- **Solution**: Used GitHub repository link for reliable dataset access

### Challenge 2: Image Preprocessing
- **Solution**: Standardized all images to 256×256 with pixel normalization

### Challenge 3: Model Overfitting
- **Solution**: Implemented data augmentation and optimized layer parameters

## Performance Optimization

### Data Pipeline Optimization:
- **cache()**: Keeps images in memory after first epoch
- **shuffle()**: Randomizes training data order for better generalization
- **prefetch()**: Prepares next batch while GPU trains current batch

## Acknowledgments

- **PlantVillage Dataset**: For providing the comprehensive potato disease dataset
- **TensorFlow/Keras Team**: For excellent deep learning framework
- **CodeBasics**: Tutorial guidance on project structure

## License

This project is open source and available under the MIT License.

## Contact

For questions, suggestions, or collaborations:
- GitHub: [@karthikkondagurla](https://github.com/karthikkondagurla)
- Email: Contact via GitHub profile

---

**Last Updated**: January 2026
**Model Version**: 1.0
