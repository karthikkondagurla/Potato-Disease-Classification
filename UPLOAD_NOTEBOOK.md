# Jupyter Notebook - Potato Disease Classification

## About the Notebook

The complete Jupyter notebook file `Potato_Disease_Classification_Complete.ipynb` contains the full implementation of the Potato Disease Classification project using TensorFlow/Keras.

## File: Potato_Disease_Classification_Complete.ipynb

This notebook has been downloaded and is ready to be added to the repository. The notebook contains the following sections:

### Contents:

1. **Data Collection Methods** - Overview of different ways to collect data
2. **Step 1: Import Libraries** - Import all necessary libraries
3. **Step 2-4: Download and Load Dataset** - Download PlantVillage dataset from GitHub
4. **Step 5: Load Dataset into TensorFlow** - Load images and create TensorFlow dataset
5. **Step 6: Visualize Sample Images** - Display sample images from each class
6. **Step 7: Data Splitting** - Split data into train/validation/test (80/10/10)
7. **Step 8: Performance Optimizations** - Apply cache, shuffle, and prefetch
8. **Step 9-11: Preprocessing and Augmentation** - Create preprocessing and augmentation layers
9. **Summary: Data Preprocessing** - Summary of preprocessing steps
10. **Model Building** - Build CNN architecture
11. **Model Compilation** - Compile model with optimizer and loss function
12. **Model Training** - Train model for 50 epochs
13. **Model Evaluation** - Evaluate on test dataset
14. **Training History Visualization** - Plot accuracy and loss curves
15. **Predictions** - Make predictions on test images
16. **Model Saving** - Save trained model with versioning
17. **Image Upload and Prediction** - Upload custom images and get predictions

## How to Use the Notebook

### Option 1: Run in Google Colab (Recommended)

1. Open the notebook in Google Colab
2. Run all cells in sequence
3. The notebook will download the dataset automatically
4. Train the model and evaluate results
5. Make predictions on your own images

### Option 2: Run Locally

1. Install dependencies: `pip install -r requirements.txt`
2. Download the notebook
3. Open with Jupyter Notebook: `jupyter notebook Potato_Disease_Classification_Complete.ipynb`
4. Run cells in sequence

## Requirements

- TensorFlow 2.19.0
- NumPy
- Matplotlib
- Pillow
- (Install using: `pip install -r requirements.txt`)

## Dataset

The notebook automatically downloads the PlantVillage dataset which contains:
- **Total Images**: 2,152
- **Classes**: 3
  - Potato Early Blight
  - Potato Late Blight
  - Potato Healthy

## Model Performance

After training, the model achieves:
- **Training Accuracy**: 92.51%
- **Validation Accuracy**: ~96%
- **Test Accuracy**: 89.66%

## Output

The notebook saves the trained model as: `saved_models/potato_model_v1.keras`

## Uploading the Notebook

To add the notebook file to this repository:

1. Go to the repository page
2. Click "Add file" > "Upload files"
3. Drag and drop `Potato_Disease_Classification_Complete.ipynb` or select it
4. Add commit message: "Add Jupyter notebook with complete implementation"
5. Commit changes

## Notes

- The notebook is designed to run in Google Colab with GPU support
- For local execution, ensure you have sufficient RAM (at least 4GB recommended)
- The dataset is ~1.5GB, so internet connection is required for download
- Training takes approximately 10-15 minutes on GPU
