# VggNet16 Implementation for CIFAR-10

This repository contains a PyTorch implementation of the VggNet16 architecture adapted for image classification on the CIFAR-10 dataset. The VggNet, originally proposed for ImageNet, explores the depth aspect of CNN architecture using small 3x3 convolutional filters. The goal is to provide a clear codebase along with an explanation of key concepts from the Vgg paper.

![VGG16 architecture](https://miro.medium.com/v2/resize:fit:850/1*B_ZaaaBg2njhp8SThjCufA.png)

## Overview

The VggNet architecture is designed to delve into the depth of Convolutional Neural Networks (CNNs) by utilizing small 3x3 convolutional filters. Unlike some previous architectures, VggNet focuses on stacking these small filters, resulting in a more expressive and discriminative decision function. The input to the network is a 224x224 RGB image, and during preprocessing, the mean RGB value is subtracted from each pixel.

The architecture includes convolutional filters of size 3x3 with a stride of 1, along with max-pooling layers of 2x2 pixels and a stride of 2. The network comprises multiple configurations with varying depths (8, 16, and 19 conv layers). Despite its increased depth, the number of weights does not surpass other contemporary shallow networks, thanks to the use of 3x3 filters. This approach can be seen as a form of regularization, as it forces the model to learn decomposed 3x3 filters with non-linearities in between, enhancing discriminative capabilities.

## Training

For training, the implementation addresses a multiclass classification problem using mini-batch gradient descent with momentum. A batch size of 256 and a momentum of 0.9 are employed. Regularization techniques include weight decay and dropout for the Multi-Layer Perceptron (MLP) layers. The network demonstrates faster convergence compared to AlexNet, attributed to implicit regularization due to greater depth and smaller convolutional filter size, along with pre-initialization of certain layers.

Weight initialization strategies are configuration-dependent. Random initialization from a normal distribution with mean zero and variance 10^-2 is used for shallower configurations. In deeper configurations, weights are initialized using those learned by shallower configurations, with intermediate layers initialized randomly. Data augmentation techniques, such as random horizontal flipping, random cropping, and color jitter, contribute to improved model generalization.

## Testing

The testing phase involves evaluating the trained VggNet16 model on the CIFAR-10 dataset. The results demonstrate the effectiveness of the implemented architecture in achieving classification accuracy. The experiments from the original Vgg paper emphasize the importance of increased network depth, the choice of convolutional filter sizes, and the benefits of ensemble learning for further reducing the error rate.

Feel free to explore the provided code and the accompanying explanation to gain insights into the VggNet architecture and its implications for image classification tasks on the CIFAR-10 dataset.

# Customizing and Training VggNet on Your Data

## Training VggNet with Custom Data

#### Configure Hyperparameters

For the training of VggNet on your own dataset, you can customize the training process by modifying the `main.py` file. This file contains the main code for dataset loading, model initialization, training, and testing loops.

(Optional) You can configure hyperparameters within the `main.py` file. Adjust parameters such as learning rate, batch size, and training epochs to suit your specific requirements.

#### Modify Dataset Handling

To experiment with different datasets or apply varied data augmentation techniques, you can make changes in the `dataset.py` file. This file handles the loading and preprocessing of the CIFAR-10 dataset by default.

#### After making changes, run the main.py script

# Additional Resources

For a deeper understanding of VggNet and its architectural choices, refer to the original paper:

- ["Very Deep Convolutional Networks for Large-Scale Image Recognition"](https://arxiv.org/abs/1409.1556) by Karen Simonyan and Andrew Zisserman.
