# Cifar10 with Cross Stage Partial Network

The Cifar10 [1] dataset is solved using Cross Stage Partial blocks, as in [2]. This was implemented in PyTorch.

## How to run

The code is written as a Jupyer Notebook. It can be run locally if [JupyterLab](https://jupyter.org/) in installed. In order to run it, just run the cells in their respective order. 

## Model

The architecture follows the model for CSPDarknet53, which is the backbone of Yolo v4 and v5. It begins with a 3x3 convolution to extract the low level features, followed by a strided convolution to downsample the feature maps by a factor of 2. Then two similar blocks are added: the Cross Stage Parial Block followed by a strided convolution (stride equal to 2). Each CSP Block consists of residual connections with a bottleneck layer (halving the feature maps), as per CSPDarknet53 architecture. Finally, an global average pooling is applied with a fully connected layer to map to the output of size 10. 

## Results

The model was trained only for 10 epochs in order to compare to our previous [work](https://github.com/vladhondru25/famous-datasets/tree/master/cifar-10). Although yielding the lowest testing accuracy (74.41%), the training and validation losses plotted below clearly show the accelerated decreasing trend, which can only suggests a longer training time i.e. a greater number of epochs will result in a better performance, with no signs of overfitting. This is not the case in our other experiments, as the plotted training and validation losses present a plateau in performance, as well as an overfitting behaviour. An cause for the performance of the CSP-based network is that due to the increased number of hidden layers, the model has more capability, and consequently the gradient decent algorithm is needed to be run for a longer period of time.

![alt text](https://github.com/vladhondru25/diverse-AI-projects/blob/master/./CrossStagePartial-Network/result.png?raw=true)

### Recommendation

If either JupyterLab is not installed on the machine or a faster training is desired, [Google Colab](https://colab.research.google.com/) can be used. It does provide free GPU for a limited amount of time, but more than sufficient to train the network.

## References

[[1](https://www.cs.toronto.edu/~kriz/cifar.html)] Cifar10 dataset \
[[2](https://arxiv.org/abs/1911.11929)] Chien-Yao Wang and Hong-Yuan Mark Liao and I-Hau Yeh and Yueh-Hua Wu and Ping-Yang Chen and Jun-Wei Hsieh. "CSPNet: A New Backbone that can Enhance Learning Capability of CNN." 2019.
