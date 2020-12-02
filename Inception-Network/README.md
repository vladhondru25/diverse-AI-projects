# Cifar10 with inception layers

The Cifar10 [1] dataset is solved using inception layers, as in [2]. This was implemented in PyTorch.

## How to run

The code is written as a Jupyer Notebook. It can be run locally if [JupyterLab](https://jupyter.org/) in installed. In order to run it, just run the cells in their respective order. 

### Architecture

### Results
An accuracy of 82.82% was achieved on the testing set. This is with approximately 5% higher than our implementation of a simple 5-layer





### Recommendation

If either JupyterLab is not installed on the machine or a faster training is desired, [Google Colab](https://colab.research.google.com/) can be used. It does provide free GPU for a limited amount of time, but more than sufficient to train ResNet18.

## References

[[1](https://www.cs.toronto.edu/~kriz/cifar.html)] Cifar10 dataset
[[2](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43022.pdf)] Christian Szegedy, Wei Liu, Yangqing Jia1, Pierre Sermanet1, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich, Google Inc., University of North Carolina, Chapel Hill, University of Michigan, Ann Arbor, Magic Leap Inc.. "Going Deeper with Convolutions." IEEE 2015.
