# 👀🤖 CNN CIFAR-10 Image Classifier

This repository contains a deep learning project that classifies images from the CIFAR-10 dataset using a Convolutional Neural Network (CNN).

## 📝 Introduction

Image classification is one of the core tasks in computer vision, and Convolutional Neural Networks (CNNs) are among the most effective tools for tackling it. This project focuses on building a CNN-based image classifier using the CIFAR-10 dataset — a widely used benchmark consisting of 60,000 32x32 color images across 10 categories.

## 🎯 Goal

The main goal of this project is to build, train and evaluate a CNN that can accurately classify CIFAR-10 images, achieving high accuracy while keeping the model architecture simple and efficient.

## 📌 Project Overview

- Load and preprocess the CIFAR-10 dataset
- Build a CNN architecture using TensorFlow
- Train the model on the training data and evaluate on the test data
- Visualize training progress through loss and accuracy curves
- Save the trained model for future inference or deployment

## 🔗 Dependencies

- Python 3.6+
- TensorFlow
- NumPy
- Matplotlib
- Seaborn

## ▶️ How to Run the Project

1. Clone the repository:

   ```shell
   git clone https://github.com/herrerovir/CNN-cifar10-image-classifier.git
   cd CNN-cifar10-image-classifier
   ```

2. Install dependencies:

   ```shell
   pip install -r requirements.txt
   ```

3. Launch Jupyter Notebook:

   ```shell
   jupyter notebook
   ```

4. Open and run the notebook `notebooks/Cnn-cifar-10-image-classifier.ipynb` to start training and evaluating the model.

## 🗂️ Repository Structure

```
CNN-cifar10-image-classifier/
│
├── model/                                   # Trained model
│   └── cnn-cifa10-model.keras
│
├── notebooks/                               # Jupyter notebooks with the full project
│   └── Cnn-cifar-10-image-classifier.ipynb
│
├── results/                                 # Model results
│   └── figures/
│       └── Confusion-matrix.jpg             # Model's confusion matrix
│       └── Training-vs-valid.jpg            # Model's accuracy and loss
│   └── model-results/
│       └── model-results.txt                # Results from the model as txt file
│
├── requirements.txt                         # Python dependencies
│
├── README.md                                # Project documentation
└── .gitignore                               # Git ignore rules
```

## 🛠️ Technical Skills

- Deep Learning fundamentals
- Convolutional Neural Networks (CNNs)
- TensorFlow and Keras for model building and training
- Data preprocessing
- Model evaluation and visualization

## 💾 Dataset

The CIFAR-10 dataset includes:

- 60,000 32x32 color images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training images and 10,000 test images

The dataset is included and automatically loaded via TensorFlow/Keras datasets API.

## 🧠 Model Architecture

The CNN model is composed of:

1. **Input:** `32x32x3` (RGB image)

2. **Conv2D (32 filters, 3x3)**
   → **Activation:** ReLU
   → **BatchNormalization**
   → **MaxPooling2D (2x2)**
   → **Dropout (0.2)**
   → **Output shape:** `15x15x32`

3. **Conv2D (64 filters, 3x3)**
   → **Activation:** ReLU
   → **BatchNormalization**
   → **MaxPooling2D (2x2)**
   → **Dropout (0.3)**
   → **Output shape:** `6x6x64`

4. **Conv2D (128 filters, 3x3)**
   → **Activation:** ReLU
   → **BatchNormalization**
   → **MaxPooling2D (2x2)**
   → **Dropout (0.4)**
   → **Output shape:** `2x2x128`

5. **Conv2D (256 filters, 3x3)**
   → **Activation:** ReLU
   → **BatchNormalization**
   → **MaxPooling2D (2x2)**
   → **Dropout (0.5)**
   → **Output shape:** `1x1x256`

6. **Flatten**
   → **Output shape:** `256`

7. **Dense (256 units)**
   → **Activation:** ReLU
   → **Dropout (0.5)**
   → **Output shape:** `256`

8. **Dense (128 units)**
   → **Activation:** ReLU
   → **Dropout (0.5)**
   → **Output shape:** `128`

9. **Dense (10 units)**
   → **Activation:** Softmax
   → **Output shape:** `10` (class probabilities)

**Loss Function:** `categorical_crossentropy`

**Optimizer:** `Adam`

**Metric:** `accuracy`

## 📊 Results

The CNN achieved a best validation accuracy of **81.89%** at epoch 58, demonstrating strong learning and generalization on the CIFAR-10 dataset. Training accuracy reached \~81.6%, closely aligning with validation and test results, indicating minimal overfitting.

Early stopping was triggered at epoch 67, with model weights restored from the best-performing epoch. The learning rate was progressively reduced during training (from `2.0e-4` to `1.56e-5`), contributing to stable convergence.

While the model performed well, slight underfitting was observed, suggesting that future improvements could include using deeper architectures, data augmentation, or further hyperparameter tuning.
