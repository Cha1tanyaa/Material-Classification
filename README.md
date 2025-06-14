# Material Classification

## 🚀 Overview
**Material Classification** is a Computer Vision project that implements a Convolutional Neural Network (CNN) to classify images of materials into different categories. It utilizes the `garythung/trashnet` dataset from Hugging Face, performs data preprocessing and augmentation, trains a model using TensorFlow/Keras, and evaluates its performance. The goal is to accurately categorize waste items to aid in recycling and waste management efforts.

## 📄 Paper
This document provides further details and context on material classification, particularly relevant to the themes explored in this project.

## ✨ Features
- **Dataset**: Leverages the [garythung/trashnet](https://huggingface.co/datasets/garythung/trashnet) dataset, a popular benchmark for material image classification.
- **Data Preprocessing**: Includes essential steps like image resizing to a uniform dimension (100x100 pixels) and normalization of pixel values.
- **Data Augmentation**: Employs various transformations such as random flips (horizontal and vertical), rotations, and brightness adjustments to artificially expand the training dataset and improve model generalization.
- **CNN Model**: Features a sequential Keras model with multiple `Conv2D` and `MaxPooling2D` layers for feature extraction, followed by `Flatten`, `Dense`, and `Dropout` layers for classification.
- **Training**: Implements an early stopping callback to prevent overfitting by monitoring validation loss and saves the best performing model.
- **Evaluation**: Assesses model performance on a dedicated test set, with visualization of training/validation accuracy and loss curves using Matplotlib.
- **Modular Code**: The project is organized into clear and distinct Python scripts for:
    - Data preparation and augmentation ([`src/data_preperation.py`](src/data_preperation.py))
    - Model definition, training, and evaluation ([`src/main.py`](src/main.py))
    - Dataset sample viewing ([`src/view.py`](src/view.py))
    - Dataset image counting ([`src/count.py`](src/count.py))
    - Testing augmentation functions ([`src/test.py`](src/test.py))

## 📂 File Structure
- [`src/`](src/) – Contains all Python source code for the project.
    - [`data_preperation.py`](src/data_preperation.py) – Downloads the dataset, preprocesses images, applies augmentation to the training set, and splits data into train/validation/test sets, saving them locally.
    - [`main.py`](src/main.py) – Defines the CNN architecture, loads the processed datasets, trains the model, evaluates it, and saves the trained model.
    - [`view.py`](src/view.py) – A utility script to load and display sample images from the augmented training dataset for visual inspection.
    - [`count.py`](src/count.py) – A utility script to count the number of images in the (augmented) training dataset.
    - [`test.py`](src/test.py) – A script designed to test specific image augmentation functions on sample images.
- [`models/`](models/) – Directory for storing trained model files.
    - [`material_classifier_model.keras`](models/material_classifier_model.keras) – The saved Keras model file after training.
- [`data/`](data/) – Directory where processed datasets are stored (this directory is typically gitignored).
    - `augmented_train_ds/` – Contains the augmented training dataset.
    - `validation_ds/` – Contains the validation dataset.
    - `test_ds/` – Contains the test dataset.
    - `train_ds/` – Contains the original, non-augmented training dataset.
- [`assets/`](assets/) – Contains static assets like images used in the README.
    - [`Figure_1.png`](assets/Figure_1.png) – The screenshot of training progress.
- [`README.md`](README.md) – This file, providing an overview and instructions for the project.
- `requirements.txt` – Lists all Python dependencies required to run the project.

## ⚙️ Setup and Installation

### Prerequisites
- Python 3.10 or newer.
- `pip` (Python package installer).
- Git (for cloning the repository).

### Installation
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Cha1tanyaa/Material-Classification.git
    cd Material-Classification
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install dependencies**:
    Ensure you have a `requirements.txt` file in the project root.
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage Workflow

The scripts are generally designed to be run from the project's root directory.

1.  **Prepare the Data**:
    This script downloads the "garythung/trashnet" dataset, preprocesses it, splits it, augments the training data, and saves the processed datasets into the `data/` directory at the project root.
    ```bash
    python src/data_preperation.py
    ```

2.  **Train the Model**:
    This script loads the preprocessed datasets, defines the CNN model, trains it, displays progress, evaluates it, and saves the trained model to `models/material_classifier_model.keras`.
    ```bash
    python src/main.py
    ```

## 🧠 Model Architecture
The CNN model ([`src/main.py`](src/main.py)) is defined as a Keras Sequential model with the following layers:

1.  **Input Layer**: Expects images of shape (100, 100, 3).
2.  **Conv2D Layer**: 32 filters, kernel size (3,3), ReLU activation.
3.  **MaxPooling2D Layer**: Pool size (2,2).
4.  **Conv2D Layer**: 64 filters, kernel size (3,3), ReLU activation.
5.  **MaxPooling2D Layer**: Pool size (2,2).
6.  **Conv2D Layer**: 128 filters, kernel size (3,3), ReLU activation.
7.  **MaxPooling2D Layer**: Pool size (2,2).
8.  **Flatten Layer**: Converts 3D feature maps to 1D feature vectors.
9.  **Dense Layer**: 256 units, ReLU activation.
10. **Dropout Layer**: Rate of 0.2 (20% dropout) to prevent overfitting.
11. **Dense Layer (Output)**: 6 units (corresponding to the 6 material categories), Softmax activation for multi-class probability distribution.

-   **Optimizer**: Adam optimizer with a learning rate of 0.001.
-   **Loss Function**: Sparse Categorical Crossentropy, suitable for integer-based class labels.
-   **Metrics**: Accuracy.

## 🛠️ Built With
-   [TensorFlow](https://www.tensorflow.org/): Core library for deep learning, model building, and training.
-   [TensorFlow Addons](https://www.tensorflow.org/addons): Used for certain image augmentation techniques (Note: TensorFlow Addons is in maintenance mode; consider alternatives for new projects).
-   [NumPy](https://numpy.org/): Fundamental package for numerical computation in Python.
-   [Hugging Face Datasets](https://huggingface.co/docs/datasets/): For easily downloading and handling the TrashNet dataset.
-   [Matplotlib](https://matplotlib.org/): For plotting training history and displaying images.

*(See `requirements.txt` for a complete list of dependencies and their versions.)*

## 📜 Contributions
This project was a collaborative effort with [Lukas](https://github.com/Lukic-sys).
