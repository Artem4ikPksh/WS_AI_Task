# MNIST Classifier Project

## Overview
This project implements three different classifiers for the **MNIST handwritten digit recognition task**:
- **Random Forest** (traditional machine learning approach)
- **Fully Connected Neural Network (NN)** (basic deep learning model)
- **Convolutional Neural Network (CNN)** (state-of-the-art image classification approach)

The design follows an **object-oriented architecture** with an abstract interface `MnistClassifierInterface`, making it easy to extend the project with new algorithms.

## Project Structure
- `MnistClassifierInterface` — abstract base class with `train` and `predict` methods.
- `RandomForestMnistClassifier` — Random Forest implementation using scikit-learn.
- `NeuralNetworkMnistClassifier` — simple feed-forward neural network using TensorFlow/Keras.
- `CnnMnistClassifier` — convolutional neural network for image classification.
- `MnistClassifier` — wrapper/facade to select algorithm (`rf`, `nn`, `cnn`).

## Setup Instructions
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/MacOS
   venv\Scripts\activate      # Windows
