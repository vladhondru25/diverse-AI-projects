# Cifar10 with inception layers

The Cifar10 [1] dataset is solved using inception layers, as in [2]. This was implemented in PyTorch.

## How to run

The code is written as a Jupyer Notebook. It can be run locally if [JupyterLab](https://jupyter.org/) in installed. In order to run it, just run the cells in their respective order. 

### Architecture
The first layer is composed of a convolution operation followed by a max pooling. The second layer consists of a convolution. Then, two inception layers are added with a max pooling in between them. The structure of the inception layer can be seen in the figure below.

![alt text](https://github.com/vladhondru25/diverse-AI-projects/blob/master/./Inception-network/inception_module.png?raw=true)

Then, an average pooling with a kernel size equal to the feature maps to result in each feature map containing only 1 value. Finally, a dropout with a rate of 50% is added before the final linear layer to map to the classes dimension. 

It is important to note that batch normalisation with rectified linear layers activation function are used after after each convolutional operation.

### Results
An accuracy of 82.82% was achieved on the testing set. This is with approximately 5% higher than our [implementation](https://github.com/vladhondru25/famous-datasets/blob/master/cifar-10/cifar10_no_resnet.ipynb) of a simple 6-layer network (5 convolutions with an average pooling in the end, followed by a linear layer). 





### Recommendation

If either JupyterLab is not installed on the machine or a faster training is desired, [Google Colab](https://colab.research.google.com/) can be used. It does provide free GPU for a limited amount of time, but more than sufficient to train ResNet18.

## References

[[1](https://www.cs.toronto.edu/~kriz/cifar.html)] Cifar10 dataset
[[2](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43022.pdf)] Christian Szegedy, Wei Liu, Yangqing Jia1, Pierre Sermanet1, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich, Google Inc., University of North Carolina, Chapel Hill, University of Michigan, Ann Arbor, Magic Leap Inc.. "Going Deeper with Convolutions." IEEE 2015.
