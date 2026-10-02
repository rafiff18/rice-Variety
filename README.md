# Rice Variety Image Classification

## Project Overview

This project implements a Convolutional Neural Network (CNN) deep learning model to accurately classify five distinct commercial varieties of rice: Arborio, Basmati, Ipsala, Jasmine, and Karacadag. Developed with TensorFlow and Keras, the pipeline covers automated dataset retrieval, balanced sampling across classes, standardized image preprocessing with aspect ratio preservation, data augmentation, hyperparameter-tuned model training with callbacks, comprehensive evaluation on an independent test set, error analysis, and multi-format deployment exports (TensorFlow SavedModel, TensorFlow Lite, and TensorFlow.js), accompanied by an interactive web inspection dashboard.

## Technology Stack

The project leverages a modern, production-grade technology stack across all stages:

### 1. Deep Learning, Modeling & Vision
- **Python (3.11 / 3.12)**: Core programming language for end-to-end model development, data pipelines, and backend serving.
- **TensorFlow 2.21.0 & Keras**: Primary framework for neural network design, custom callbacks, loss functions, and optimization routines.
- **NumPy (2.4.3 / 1.26.4)**: High-performance multidimensional array manipulation and numerical tensor operations.
- **Pillow (PIL)**: Image decoding, boundary calculations for aspect-ratio preservation, and bilinear spatial resizing.
- **Scikit-Learn (1.9.0)**: Exhaustive evaluation diagnostics, generating precision, recall, F1-scores, and multiclass confusion matrix calculations.
- **Matplotlib (3.10.8)**: Visualization of training/validation loss, learning curves, and exploratory class distribution plots.

### 2. Data Engineering & Pipeline Optimization
- **kagglehub (1.0.0)**: Headless, programmatic retrieval and version-controlled caching of the Kaggle Rice Image Dataset.
- **tf.data Pipeline**: Asynchronous data loading utilizing memory caching (`cache()`), random buffer shuffling, and thread prefetching (`AUTOTUNE`) for maximized hardware utilization.
- **Keras Data Augmentation**: In-graph stochastic transformations including horizontal/vertical flipping, rotation, and zooming.

### 3. Model Compression & Edge Deployment Formats
- **TensorFlow Lite (TFLite)**: Flatbuffer serialization (`tflite/model.tflite`, 11.06 MB) with XNNPACK runtime delegate for low-latency (< 15 ms) CPU/mobile inference.
- **TensorFlow SavedModel**: Full protocol-buffer graph representation (`saved_model/`) with complete computation graph and signature definitions for cloud containerized deployments.
- **TensorFlow.js (TFJS)**: Sharded binary topology and weights (`tfjs_model/`) enabling zero-server client-side inference directly inside web browsers.

### 4. Web Application & Backend API
- **FastAPI (0.141.1)**: Asynchronous, OpenAPI-compliant Python web framework providing high-throughput inference endpoints.
- **Uvicorn (0.52.1)**: Production-grade ASGI web server implementation based on uvloop and httptools.
- **python-multipart**: Streaming parser for multipart/form-data image uploads.

### 5. Frontend & Interface Engineering
- **HTML5**: Semantic web architecture with native media stream integration (`<video>`, `<canvas>`) for live camera capture.
- **Vanilla CSS3**: Tailored dark-mode design system featuring backdrop blur (glassmorphism), responsive CSS Grid, custom glow tokens, and fluid micro-animations without external UI library dependencies.
- **Vanilla JavaScript (ES6+)**: Asynchronous Fetch API client, live preview rendering, 1-click sample dataset test triggers, and animated probability distribution indicators.
- **Google Fonts (Outfit & Plus Jakarta Sans)**: High-legibility modern typography system.

## Dataset

The dataset utilized in this project is the **Rice Image Dataset** created by Murat Koklu, sourced from Kaggle.
- Source identifier: `muratkokludataset/rice-image-dataset` (retrieved via `kagglehub`)
- Total dataset size: 10,000 images (2,000 images per class)
- Class balance: Perfectly balanced distribution across all 5 classes (2,000 images each, representing exactly 20.0% per class)
  - Arborio: 2,000 images
  - Basmati: 2,000 images
  - Ipsala: 2,000 images
  - Jasmine: 2,000 images
  - Karacadag: 2,000 images
- Original image resolution: 250x250 pixels (RGB)
- Standardized image resolution: 150x150 pixels with 3 color channels (RGB)
- Dataset partition:
  - Training set: 80% (8,000 images, 250 batches of size 32)
  - Validation set: 10% (992 images, 31 batches of size 32)
  - Test set: 10% (1,008 images, 32 batches of size 32)
  - Random seed: 123

## Preprocessing and Data Augmentation

To address variations in resolution and orientation while preventing distortion of grain morphology:
1. Aspect Ratio Preservation: Image loading uses `crop_to_aspect_ratio=True` inside `tf.keras.utils.image_dataset_from_directory`. This crops the input to the target aspect ratio prior to bilinear resizing, ensuring that elongation and roundness of the rice grains are preserved.
2. Pixel Normalization: Pixel intensities are rescaled from [0, 255] to [0, 1] using a `Rescaling(1./255)` layer.
3. Data Augmentation: Applied dynamically during training via Keras layers:
   - Random horizontal and vertical flips (`RandomFlip("horizontal_and_vertical")`)
   - Random rotations up to 15% (`RandomRotation(0.15)`)
   - Random zoom up to 15% (`RandomZoom(0.15)`)
4. Pipeline Optimization: Datasets are cached in memory and prefetched using `tf.data.AUTOTUNE` to maximize GPU/CPU throughput.

## Model Architecture

The model is built using the Keras Sequential API with the following layers:
1. Input and Rescaling: Accepts (150, 150, 3) tensors and rescales pixel values to [0, 1].
2. Data Augmentation: Applied only during training phase.
3. Convolutional Feature Extractor:
   - Block 1: Conv2D (32 filters, 3x3 kernel, ReLU activation, same padding) + MaxPooling2D (2x2 pool size)
   - Block 2: Conv2D (64 filters, 3x3 kernel, ReLU activation, same padding) + MaxPooling2D (2x2 pool size)
   - Block 3: Conv2D (128 filters, 3x3 kernel, ReLU activation, same padding) + MaxPooling2D (2x2 pool size)
   - Block 4: Conv2D (128 filters, 3x3 kernel, ReLU activation, same padding) + MaxPooling2D (2x2 pool size)
4. Classification Head:
   - Flatten layer
   - Dropout layer (rate 0.4) for regularization
   - Dense layer (256 units, ReLU activation)
   - Dense output layer (5 units, Softmax activation)
5. Compilation:
   - Optimizer: Adam with tuned learning rate of `0.0003` for smooth convergence
   - Loss Function: Sparse Categorical Crossentropy
   - Metrics: Accuracy
6. Training Callbacks:
   - `TargetCallback`: Halts training when both training accuracy and validation accuracy reach >= 95%.
   - `EarlyStopping`: Monitors validation loss with a patience of 4 epochs and restores best weights.
   - `ReduceLROnPlateau`: Reduces learning rate by a factor of 0.5 if validation loss stagnates for 1 epoch (minimum learning rate: 1e-5).

## Evaluation Results

The model met the early stopping target criteria at Epoch 5:
- Final Training Accuracy: 95.80% (Loss: 0.1176)
- Final Validation Accuracy: 96.47% (Loss: 0.0971)
- Test Set Accuracy: 97.52% (Loss: 0.0753)

Both training and testing accuracy exceed the minimum threshold of 95%.

### Detailed Classification Metrics (Test Set, 1,008 Samples)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| Arborio | 0.9571 | 0.9617 | 0.9594 | 209 |
| Basmati | 0.9679 | 1.0000 | 0.9837 | 211 |
| Ipsala | 0.9948 | 1.0000 | 0.9974 | 193 |
| Jasmine | 0.9950 | 0.9617 | 0.9781 | 209 |
| Karacadag | 0.9620 | 0.9516 | 0.9568 | 186 |
| **Overall Accuracy** | | | **0.9752** | **1,008** |
| **Macro Average** | **0.9754** | **0.9750** | **0.9751** | **1,008** |
| **Weighted Average** | **0.9754** | **0.9752** | **0.9752** | **1,008** |

### Confusion Matrix

| Actual \ Predicted | Arborio | Basmati | Ipsala | Jasmine | Karacadag |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Arborio** | 201 | 0 | 0 | 1 | 7 |
| **Basmati** | 0 | 211 | 0 | 0 | 0 |
| **Ipsala** | 0 | 0 | 193 | 0 | 0 |
| **Jasmine** | 0 | 7 | 1 | 201 | 0 |
| **Karacadag** | 9 | 0 | 0 | 0 | 177 |

### Error Analysis

The confusion matrix indicates strong class separability across varieties:
1. Basmati and Ipsala achieved 100.00% recall with zero false negatives due to distinctive elongated slender profiles and large grain outlines.
2. Slight confusion occurred between Arborio and Karacadag (7 Arborio predicted as Karacadag, 9 Karacadag predicted as Arborio) due to shared round-to-medium chalky grain characteristics.
3. Minor misclassification between Jasmine and Basmati (7 samples) resulted from overlapping length-to-width ratios in specific grain orientations.
4. Overall, macro and weighted F1-scores both reached 97.51% and 97.52%, confirming balanced precision and recall across all classes without bias toward majority patterns.

## How to Run

### 1. Requirements and Environment Setup

Ensure Python (version 3.9 to 3.12) is installed. Install all project dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies in `requirements.txt`:
- `tensorflow==2.21.0`
- `kagglehub==1.0.0`
- `matplotlib==3.10.8`
- `numpy==2.4.3`
- `scikit-learn>=1.3.0`
- `tensorflowjs`

### 2. Running Training in Google Colab or Jupyter Notebook

1. Open `notebook.ipynb` in Google Colab (recommended with T4 GPU runtime) or a local Jupyter Notebook environment.
2. Run all cells sequentially:
   - Cell 1-4: Install `kagglehub`, `tensorflowjs`, and import core libraries.
   - Cell 5-9: Automatically download the Kaggle dataset, balance 2,000 images per class (10,000 total images), verify distribution, and plot sample images.
   - Cell 10-13: Preprocess dataset using `crop_to_aspect_ratio=True` and partition into training (80%), validation (10%), and testing (10%) splits.
   - Cell 14-15: Construct CNN model, compile with Adam (learning rate 0.0003), and fit model with `TargetCallback` and `EarlyStopping`.
   - Cell 16-17: Evaluate model on the test set, output classification report, and generate accuracy and loss curves.
   - Cell 18-20: Export model to TensorFlow SavedModel (`saved_model`), TensorFlow Lite (`tflite/model.tflite`), and TensorFlow.js (`tfjs_model`), then generate `submission.zip`.

### 3. Running TFLite Inference

To run inference on a sample image using the exported TensorFlow Lite model:

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

# Select a test image
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

# Parse prediction
pred_idx = np.argmax(output_data[0])
confidence = np.max(output_data[0]) * 100

print(f"Test Image: {img_path}")
print(f"Predicted Class: {class_names[pred_idx]}")
print(f"Confidence: {confidence:.2f}%")
```

### 4. Running the Interactive Web Application

A full-featured web inspection dashboard powered by FastAPI and the TFLite edge engine is included in the `web_app/` directory:

1. Navigate to the `web_app` directory:
   ```bash
   cd web_app
   ```
2. Start the FastAPI server using Uvicorn:
   ```bash
   uvicorn app:app --host 127.0.0.1 --port 8000 --reload
   ```
3. Open your web browser and navigate to:
   ```
   http://127.0.0.1:8000
   ```
4. Features available in the web interface:
   - Drag & drop rice photo upload or live webcam grain capture.
   - 1-Click instant test chips loaded from the verified dataset.
   - Real-time probability bar chart for all 5 varieties.
   - Agronomic and culinary metadata cards (origin, texture, culinary uses, cooking water ratio, glycemic index).
   - Low-latency inference reporting (< 15 ms).

## Conclusion

1. Scaled and Balanced Dataset: The dataset was expanded to 10,000 images with an equal distribution of 2,000 images per class, satisfying dataset size criteria and preventing class bias.
2. Standardized Preprocessing: Standardized 150x150 resolution using `crop_to_aspect_ratio=True` and pixel rescaling preserved grain geometric properties and enhanced feature consistency.
3. Exceeded Accuracy Targets: Both training accuracy (95.80%) and test accuracy (97.52%) comfortably exceeded the 95% minimum requirement, supported by a macro F1-score of 97.51% and weighted F1-score of 97.52%.
4. Multi-Platform Deployment: The model was successfully converted to TensorFlow SavedModel, TensorFlow Lite (`tflite/model.tflite`, 11.06 MB), and TensorFlow.js (`tfjs_model`), ready for web and mobile edge inference.
