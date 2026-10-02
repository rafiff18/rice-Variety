# Rice Variety Image Classification

## Project Overview

This project implements a Convolutional Neural Network (CNN) deep learning model to classify five distinct varieties of rice: Arborio, Basmati, Ipsala, Jasmine, and Karacadag. Developed with TensorFlow and Keras, the pipeline covers automated dataset retrieval, image preprocessing, data augmentation, model training with early stopping, model evaluation, and multi-format deployment exports (TensorFlow SavedModel, TensorFlow Lite, and TensorFlow.js).

## Dataset

The dataset utilized in this project is the **Rice Image Dataset** created by Murat Koklu, sourced from Kaggle.
- Source identifier: `muratkokludataset/rice-image-dataset` (retrieved via `kagglehub`)
- Class distribution: 300 samples per class across 5 varieties (total: 1,500 images)
  - Arborio: 300 images
  - Basmati: 300 images
  - Ipsala: 300 images
  - Jasmine: 300 images
  - Karacadag: 300 images
- Image resolution: Resized to 150x150 pixels with 3 color channels (RGB)
- Dataset split:
  - Training set: 80% (1,200 images, 38 batches)
  - Validation set: 10% (160 images, 5 batches)
  - Test set: 10% (140 images, 5 batches)
  - Random seed: 123

## Model Architecture

The neural network is built using the Keras Sequential API with the following structure:
1. Rescaling layer: Normalizes pixel values from [0, 255] to [0, 1].
2. Data augmentation layers: Random horizontal/vertical flip, random rotation (factor 0.2), and random zoom (factor 0.2).
3. Convolutional blocks: Four consecutive blocks of Conv2D (3x3 kernel, ReLU activation, same padding) followed by MaxPooling2D (2x2 pool size) with filter depths of 32, 64, 128, and 128.
4. Dense classification head: Flatten layer, Dropout layer (rate 0.5), Dense layer (512 units, ReLU activation), and a final Dense output layer (5 units, Softmax activation).
5. Optimization: Compiled with Adam optimizer and Sparse Categorical Crossentropy loss.
6. Callbacks:
   - `TargetCallback`: Automatically halts training when training accuracy >= 95% and validation accuracy >= 95%.
   - `EarlyStopping`: Monitors validation loss with a patience of 10 epochs and restores best weights.

## Evaluation Results

Training stopped at Epoch 23 when the target callback criteria were fulfilled:
- Final Training Accuracy: 97.25% (Loss: 0.0751)
- Final Validation Accuracy: 96.88% (Loss: 0.1566)
- Notebook Test Evaluation: Accuracy 92.86% (Loss: 0.2114)

### Detailed Classification Metrics (Test Set)

Below is the classification report evaluated on the test set:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| Arborio | 1.0000 | 0.9000 | 0.9474 | 20 |
| Basmati | 0.9688 | 1.0000 | 0.9841 | 31 |
| Ipsala | 0.9259 | 1.0000 | 0.9615 | 25 |
| Jasmine | 1.0000 | 0.9714 | 0.9855 | 35 |
| Karacadag | 1.0000 | 1.0000 | 1.0000 | 29 |
| **Accuracy** | | | **0.9786** | 140 |
| **Macro Average** | **0.9789** | **0.9743** | **0.9757** | 140 |
| **Weighted Average** | **0.9799** | **0.9786** | **0.9785** | 140 |

### Confusion Matrix

| Actual \ Predicted | Arborio | Basmati | Ipsala | Jasmine | Karacadag |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Arborio** | 18 | 0 | 2 | 0 | 0 |
| **Basmati** | 0 | 31 | 0 | 0 | 0 |
| **Ipsala** | 0 | 0 | 25 | 0 | 0 |
| **Jasmine** | 0 | 1 | 0 | 34 | 0 |
| **Karacadag** | 0 | 0 | 0 | 0 | 29 |

## How to Run

### 1. Requirements and Setup

Ensure Python (3.9 to 3.12) is installed. Install all project dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies from `requirements.txt`:
- `tensorflow==2.21.0`
- `kagglehub==1.0.0`
- `matplotlib==3.10.8`
- `numpy==2.4.3`
- `tensorflowjs` (for web model conversion)

### 2. Running Training in Google Colab or Jupyter Notebook

1. Open `notebook.ipynb` in Google Colab or your local Jupyter environment.
2. Execute the notebook cells sequentially:
   - Cell 1-4: Dependency installations and library imports.
   - Cell 5-9: Automated dataset download from Kaggle (`kagglehub`) into `./rice-dataset`, copying 300 samples per class, and visualization of class distribution.
   - Cell 10-13: Dataset batch preparation (batch size 32, image resolution 150x150) and dataset partitioning into 80% train, 10% validation, and 10% test splits.
   - Cell 14-15: Model architecture declaration, compilation, and training with `TargetCallback` and `EarlyStopping`.
   - Cell 16-17: Test set evaluation and generation of training/validation loss and accuracy curves.
   - Cell 18-20: Model export to TensorFlow SavedModel (`submission/saved_model`), TensorFlow Lite (`submission/tflite/model.tflite`), and TensorFlow.js (`submission/tfjs_model`), along with zip compression.

### 3. Running TFLite Inference

To run inference on a single test image using the exported TensorFlow Lite model:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pathlib
import random

# Load labels
with open("tflite/label.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Initialize TFLite interpreter
interpreter = tf.lite.Interpreter(model_path="tflite/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Select a sample image
sample_images = list(pathlib.Path("./rice-dataset").rglob("*.jpg"))
img_path = random.choice(sample_images)

# Preprocess image
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Invoke interpreter
interpreter.set_tensor(input_details[0]["index"], img_array)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]["index"])

# Get prediction
pred_idx = np.argmax(output_data[0])
confidence = np.max(output_data[0]) * 100

print(f"Predicted Class: {class_names[pred_idx]}")
print(f"Confidence: {confidence:.2f}%")
```

## Conclusion

1. High Classification Performance: The CNN model achieved robust performance across all five rice varieties, achieving an overall test accuracy of 92.86% in the notebook evaluation and 97.86% on the exported inference model, with a macro average F1-score of 0.9757 and weighted average F1-score of 0.9785.
2. Fast Convergence and Generalization: With the combination of convolutional feature extractors, data augmentation, dropout regularization, and custom callbacks, the model surpassed the 95% accuracy requirement within 23 epochs without noticeable overfitting.
3. Multi-Platform Deployment: The model was successfully converted and verified across multiple formats (TensorFlow SavedModel, TensorFlow Lite, and TensorFlow.js), enabling immediate deployment on mobile, edge, and browser environments.
