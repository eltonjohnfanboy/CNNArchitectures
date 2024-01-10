In this repo I will be implementing the AlexNet network, designed by **Alex Krizhevsky** in collaboration with **Ilya Sutskever** and **Geoffrey Hinton**. 

Before talking about the whole model, and it's architecture, I would like to mention four of the big/important features that were used in it, some of which are still used nowadays in modern architectures (we will talk about these in more detail later):
- The use of ReLU non-linearity
- Overlap pooling to reduce the size of the network
- Use dropout to combat overfitting issues 
- Data augmentation<br>


First, let's start with the architecture of AlexNet, which consists of 8 layers with weights. 5 of these eight layers are convolutional layers and 3 fully connected layers. For each of these layers they use the ReLU activation function (except for the last/output layer which uses a softmax), also they make use of dropout in the first two fully connected layers. Furthermore, overlapping max-pooling is also applied after the 1st, 2nd and 5th convolutional layer.  
You can see the architecture visualized in the following image:
![source: https://medium.com/analytics-vidhya/concept-of-alexnet-convolutional-neural-network-6e73b4f9ee30](https://miro.medium.com/max/960/0*pJ3o_2zTTNnixhKH.png?) 

It's also important to mention that the time when this model was first introduced, the GPUs used to have around 3 GB of memory only, this made the training process way harder. AlexNet specifically, was trained between five and six days to train on two GTX 580 GPUs with 3GB memory. That's why, if we look at the AlexNet paper, we can see how they explain that a multi-GPU training was used by having half of the model in one GPU and the other half on another GPU with occasional intercommunication. The researchers mention in the paper that they believed that the results can be improved if faster GPUs and bigger datasets are used (which nowadays we know that it is indeed true). 

Now let's talk about each of the features mentioned before, and how they affected the model, and it's performance.
1. **Use of ReLU non-linearity:**
Even though the ReLU non-linearity funciton already existed, before AlexNet we usually used Sigmoid and tanh as activation function, which we know can be problematic (vanishing gradient problem). AlexNet makes use of the ReLU non-linearity, which gives a better performance than the other activation functions: avoiding vanishing gradients and having a better/faster convergence performance. We can see the comparison between the tanh function and the ReLU function in the following image:
<img src="https://deeplearning.vn/post/alexnet/images/relu.png" width="350"> <br>
2. **Overlapping pooling:** Instead of using a traditional pooling layer which uses a 2x2 stride, they use a pooling layer of 3x3 filter/window with 2x2 stride, so there's always pixels that are overlapped. In the paper they mention how using this strategy they are able to reduce the top-1 and top-5 error rate by 0.4% and 0.3% respectively. Still, this strategy is not really used nowadays.
3. **Dropout:** Basically consists of turning off neurons with a predetermined probability in different iterations, so they don't contribute to the forward pass and do not participate in backpropagation. It's a regularization technique that the authors used to avoid the overfitting that the model was presenting.
4. **Data augmentation:** Again, this was a technique that was already known, and it's commonly used to reduce overfitting on image data. In this case, the authors generate images with translations and horizontal reflections (random cropping), which increases the training set by a factor of 2048 (these data augmentation technique are still used today in computer vision problems and are highly effective). Furthermore, a form of color jittering is also used, altering the intensities of the RBG channels in training images (performing PCA on the RBG pixel values). <br>
It's worth mentioning that in the paper, the authors emphasize that both the Data augmentation and the Dropout strategies were used specially because they wanted to reduce the model's overfitting problem. 

**Problems that AlexNet faces:**
- Lacks depth, fails to compete with later models like VGGNet or ResNet. In fact, one of the main differences between AlexNet and VGGNet is the depth of the model. On the other hand, what ResNet improves with respect to the previous models is the skip connection strategy.
- The use of 5x5 conv filters. It's not really used nowadays, it's preferable to use two 3x3 filters for example, since it's equivalent to one 5x5 filter but with the advantage that it reduces the number of weights/parameters used and gets us more non-linearity.  
- The weight initialization. The authors make use of a normal distribution for the weight initialization of AlexNet, which doesn't really solves the vanishing gradient problem, nowadays we know that it's better to make use of the Xavier or Kiaming initialization. 

Finally, it's also relevant to bring up that Alexnet was one of the first CNN architecture used in the ImageNet challenge (2012), and it ended up winning the challenge, achieving a top-5 error rate of 15.3%, which was 10.8% lower than the runner-up. <br>
You can check the original paper of this architecture using the following link: https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
