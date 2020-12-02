# Cifar10

The implementation of ResNet18 [1] in PyTorch. The residual neural network with 18 layers is trained on Cifar10 [2] dataset. Furthermore, many networks that do not use residual layers are tried to solve the Cifar10 dataset. 

## How to run

The code is written as a Jupyer Notebook. It can be run locally if [JupyterLab](https://jupyter.org/) in installed. In order to run it, just run the cells in their respective order. 

## Experiments for solving Cifar10 without residual layers

### Kernel size
File: "cifar10_no_resnet_kernel_size.ipynb":
The experiment was related to the size of the kernel size, with a comparison between sizes of 3 vs 5. The result was a higher accuracy was yieleded by the architecture with kernels of size 5. This can be mainly attributed to the first layer, which is able to extract features maps of greater importance.

### Max Pooling vs Strided convolutions
File "cifar10_no_resnet_pool_vs_sconv.ipynb":
The experiement tested the peformance of using strided convolutions against max pooling for the downsampling method. It resulted in a better performance when using max pooling. A potential reason is that the due to the limited training time (only 10 epochs), the strided convolutions did not converge to a local optimal solution, whereas max pooling is a fixed operation. 

### Dropout
File "cifar10_no_resnet_dropout.ipynb": 
The experiment assessed the impact of adding Dropout after each activation function in each convolution layer. It concluded in a lower accuracy, however, this can be expected, as due to the Dropout, the network requires more time to be trained. 



### Recommendation

If either JupyterLab is not installed on the machine or a faster training is desired, [Google Colab](https://colab.research.google.com/) can be used. It does provide free GPU for a limited amount of time, but more than sufficient to train ResNet18.

## References

[[1](https://arxiv.org/pdf/1512.03385.pdf)] Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun. "Deep Residual Learning for Image Recognition." arXiv, 1512.03385, 2015.
[[2](https://www.cs.toronto.edu/~kriz/cifar.html)] Cifar10 dataset
