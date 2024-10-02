# MNIST Digit Classification - Deep Learning with Keras and TensorFlow 2.0

This project implements a deep learning model to classify handwritten digits from the MNIST dataset using Keras and TensorFlow 2.0. The MNIST dataset contains 70,000 images of handwritten digits (0-9) and is a benchmark dataset for image classification tasks.

## Descripton 🔢

The project involves loading the dataset, preprocessing the images, building and training a neural network, and evaluating the model's performance. This README outlines the key components of the project, including setup, execution, and results.

#### 1. Data Collection

- The MNIST dataset is directly available through the Keras library.
- The dataset consists of 60,000 training images and 10,000 test images, each of size 28x28 pixels.

#### 2. Data Preprocessing

- The images are normalized by scaling pixel values to a range of [0, 1].
- The labels are converted to one-hot encoding for multi-class classification.

#### 3. Model Development

- Built a deep neural network with the following architecture:
  - Input layer: 784 nodes (28x28 pixels flattened).
  - Hidden layers: Two dense layers with 256 and 128 neurons, ReLU activation.
  - Output layer: 10 neurons (one for each digit), with softmax activation for classification.

#### 4. Model Training

- Used the Adam optimizer and categorical crossentropy as the loss function.
- Trained the model for 10 epochs with a batch size of 32.

#### 5. Model Evaluation

- Visualized the val_accuracy , accuracy , loss and val_loss using Matplotlib
- Implemented the model and tested on test data.

## Limitations ⛔️

- The current model is a simple fully connected neural network. Performance may improve by using Convolutional Neural Networks (CNNs)
- Dataset limited to grayscale images with a fixed resolution of 28x28 pixels.

## Code Requirements 📱

To run this project, you need to install the following dependencies:

- TensorFlow 2.0
- Keras
- NumPy
- Matplotlib
- Scikit-learn

```bash
  pip install tensorflow==2.0 numpy matplotlib scikit-learn
```

## Execution ▶️

Click on Run all button in the Jupyter Notebook file

## Results 📈

![Test accuracy](https://github.com/SahilTuladhar/MNIST-Digit-Classification/blob/master/images/acc.png)

![Test loss](https://github.com/SahilTuladhar/MNIST-Digit-Classification/blob/master/images/loss.png)

![Real Test on Data](https://github.com/SahilTuladhar/MNIST-Digit-Classification/blob/master/images/output.png)
