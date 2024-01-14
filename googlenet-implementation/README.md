# GoogLeNet (Inception) Implementation for CIFAR-10

## Introduction
This repository contains an implementation of GoogLeNet, a deep neural network architecture with 22 layers, designed for improved performance through increased depth and efficiency. The key innovation in GoogLeNet is the use of the Inception module, a building block that enables the network to go deeper while mitigating computational challenges.

## Motivation and High-Level Consideration
GoogLeNet addresses the challenges of improving deep neural network (NN) performance by increasing both depth and width. However, this approach introduces issues such as overfitting and increased computational resource requirements. To tackle these challenges, the authors propose sparsely connected architectures, leading to the development of the Inception module. This architecture is particularly beneficial for localization and object detection tasks.

## Architecture

![GoogLeNet architecture](https://miro.medium.com/v2/resize:fit:1400/0*G47uhQi2slwZI9-o.png)

### Inception Module
The Inception module aims to approximate optimal local sparse structures in CNNs using readily available dense components. It consists of parallel convolutions with different filter sizes, but to manage computational costs, 1x1 convolutional layers are added before larger filters. These 1x1 layers reduce the number of channels and introduce non-linearity through ReLU activation functions. Inception modules are strategically placed in the network, with the option to use them primarily in higher layers for efficiency.

### Network in Network (NiN) Approach
GoogLeNet incorporates the NiN approach, involving 1x1 convolutions and ReLU activation functions to increase the representational power of neural networks. The 1x1 convolutions serve two main purposes: dimension reduction to remove computational bottlenecks and increasing the network's width without a significant performance penalty.

### Addressing Vanishing Gradient Problem
Given the network's considerable depth, GoogLeNet addresses the vanishing gradient problem by introducing auxiliary classifiers connected to intermediate layers. These classifiers provide additional regularization and facilitate the flow of gradients to lower stages of the model.

## Training
GoogLeNet was trained using stochastic gradient descent with a momentum of 0.9. The training employed a fixed learning rate schedule that decreased the learning rate by a factor of 4 every 8 epochs. Furthermore the auxiliary classifiers are only used during this training phase and their loss is added to the total loss with a discount weight (0.3 for each one).

## Using the Code

### Training GoogLeNet with Custom Data

1. **Configure Hyperparameters:**
   For training GoogLeNet on your own dataset, customize the training process by modifying the `main.py` file. Adjust parameters such as learning rate, batch size, and training epochs to suit your specific requirements.

   (Optional) You can configure hyperparameters within the `main.py` file.

2. **Modify Dataset Handling:**
   To experiment with different datasets or apply varied data augmentation techniques, make changes in the `dataset.py` file. This file handles the loading and preprocessing of the dataset by default.

3. **Run the Training Script:**
   After making changes, run the `main.py` script to initiate the training process.

### Additional Resources

For a deeper understanding of GoogLeNet and its architectural choices, refer to the original paper:

- ["Going Deeper with Convolutions"](https://arxiv.org/abs/1409.4842) by Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.

Feel free to explore and adapt this implementation for your specific projects or research endeavors.
