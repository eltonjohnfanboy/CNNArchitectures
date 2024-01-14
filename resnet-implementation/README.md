# ResNet Implementation in PyTorch for CIFAR-10

This repository contains the code for implementing ResNet in PyTorch, applied to the CIFAR-10 dataset. The implementation follows the concepts presented in the ResNet paper titled "Deep Residual Learning for Image Recognition" by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.

![ResNet architecture](https://www.baeldung.com/wp-content/uploads/sites/4/2022/11/resnet_18_34-e1667587575245.png)

## Introduction

The authors emphasize the importance of depth in neural networks for achieving state-of-the-art results in various tasks. While issues like vanishing/exploding gradients have been mitigated with regularization techniques, a new problem arises: as network depth increases, accuracy saturates and then degrades. This degradation is not due to overfitting, and adding more layers only exacerbates the issue.

To tackle this, the authors propose a deep residual learning framework. They introduce explicit connections to deeper networks by adding shortcuts to previous shallower layers. These shortcuts perform identity mapping, aiming to make optimization easier. Despite the additional layers, this approach does not introduce extra parameters or computational complexity.

## Deep Residual Learning

The key idea is to have the stacked layers approximate a residual function (H(x) - x) instead of directly approximating H(x). By adding a skip connection, the network aims to learn the residual function, making it easier for solvers to optimize. Residual blocks, consisting of convolutional layers with BatchNorm and ReLU activations, utilize shortcut connections to achieve this. If the dimensions of x and F(x) differ, a linear projection W is applied to x.

## Network Architecture

The baseline architecture is inspired by VGG nets, with 3x3 convolutions and specific design rules. Downsampling is achieved using convolutional layers with a stride of 2. Skip connections are directly used when input and output dimensions match. When dimensions differ, two options are considered: identity mapping with zero-padding or 1x1 convolutions to match dimensions.

The implementation involves standard practices such as image resizing, random cropping, mean subtraction, color jittering, BatchNorm, Kaiming weight initialization, and an AGD optimizer. No dropout is used, and learning rate scheduling is applied.

## Experiments

Experiments compare plain networks (without skip connections) and ResNets with skip connections. The degradation problem is evident in plain networks, where deeper networks perform worse. In contrast, ResNets not only prevent degradation but also converge faster, demonstrating the effectiveness of the proposed framework. The authors rule out the vanishing gradient problem as the cause, as plain networks use BatchNorm to ensure non-zero variances in forward propagated signals.

## Using the Code

### Training ResNet with Custom Data

1. **Configure Hyperparameters:**
   For training GoogLeNet on your own dataset, customize the training process by modifying the `main.py` file. Adjust parameters such as learning rate, batch size, and training epochs to suit your specific requirements.

   (Optional) You can configure hyperparameters within the `main.py` file.

2. **Modify Dataset Handling:**
   To experiment with different datasets or apply varied data augmentation techniques, make changes in the `dataset.py` file. This file handles the loading and preprocessing of the dataset by default.

3. **Run the Training Script:**
   After making changes, run the `main.py` script to initiate the training process.

### Additional Resources

For a comprehensive understanding of ResNet and the underlying concepts, we recommend reading the original paper:

- ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

Feel free to explore and adapt this implementation of ResNet in PyTorch for the CIFAR-10 dataset for your specific projects, educational purposes, or research endeavors.
