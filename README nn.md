# Object Classification using Transfer Learning

## Project Overview

This project focuses on image classification using Convolutional Neural Networks (CNNs) and ResNet50 models, with a primary emphasis on transfer learning. By utilizing pre-trained ResNet50 weights from ImageNet, we aim to enhance model performance for object classification, specifically when limited data and computing resources are available. The project demonstrates the effectiveness of transfer learning compared to building models from scratch.

## Objectives

The primary objectives of the project are:
- To evaluate the impact of transfer learning on object classification tasks.
- To compare model performance between a CNN built from scratch, a ResNet50 model trained from scratch, and a ResNet50 model using pre-trained weights from ImageNet.
- To demonstrate that transfer learning improves accuracy, precision, and recall when limited data is available.

## Models

1. **CNN Model from Scratch**: Built using multiple convolutional layers and fully connected layers.
2. **ResNet50 from Scratch**: A ResNet50 model initialized with random weights, trained specifically on the dataset.
3. **ResNet50 with Transfer Learning**: A ResNet50 model initialized with pre-trained ImageNet weights, then fine-tuned on our dataset.

## Dataset

The project uses the **CIFAR-10** dataset, which consists of:
- 60,000 32x32 color images across 10 classes (automobile, airplane, bird, cat, deer, dog, frog, horse, ship, truck).
- 50,000 training images and 10,000 test images.

The dataset is split into training, validation, and test sets for model evaluation.

## Approach

### Data Preprocessing
- **Normalization**: Image pixel values were normalized to a [0, 1] range to ensure consistent and efficient training.
- **One-Hot Encoding**: Target labels were one-hot encoded for multi-class classification tasks.

### Model Training
Each model was trained for 50 epochs with a batch size of 64. We used **Adam optimizer** with a learning rate of 0.0001 and **categorical cross-entropy** as the loss function. Evaluation metrics include accuracy, precision, and recall.

### Model Architectures
- **CNN from Scratch**: Consists of convolutional layers with increasing filters, batch normalization, max pooling, fully connected layers, and dropout for overfitting prevention.
- **ResNet50 from Scratch**: Built from the ground up without pre-trained weights, employing a residual learning framework.
- **ResNet50 Transfer Learning**: Utilizes pre-trained ImageNet weights for faster convergence and better performance with smaller datasets.

## Results

### Model Performance
| Model                  | Test Loss | Test Accuracy | Test Precision | Test Recall |
|------------------------|-----------|---------------|----------------|-------------|
| CNN from Scratch        | 1.58      | 70%           | 0.71           | 0.69        |
| ResNet50 from Scratch   | 1.56      | 66%           | 0.68           | 0.65        |
| ResNet50 Transfer Learning | 0.52   | 88%           | 0.88           | 0.87        |

The **ResNet50 Transfer Learning** model outperformed both the CNN and ResNet50 models trained from scratch, achieving the highest accuracy (88%) and precision (0.88), and exhibiting minimal overfitting.

### Key Insights
- Transfer learning significantly improves classification performance, especially when working with limited data.
- Pre-trained ResNet50 converged faster and achieved higher accuracy compared to models trained from scratch.

## Conclusion

This study demonstrates the power of transfer learning in object classification tasks. By utilizing pre-trained models like ResNet50, it is possible to achieve superior performance even with limited data and computational resources. The results suggest that transfer learning is a highly efficient approach for deep learning applications.


