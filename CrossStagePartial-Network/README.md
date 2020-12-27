# Cifar10 with Cross Stage Partial Network

The Cifar10 [1] dataset is solved using Cross Stage Partial blocks, as in [2]. The architecture follows the model for CSPDarknet53, which is the backbone of Yolo v4 and v5. This was implemented in PyTorch.

## How to run

The code is written as a Jupyer Notebook. It can be run locally if [JupyterLab](https://jupyter.org/) in installed. In order to run it, just run the cells in their respective order. 


### Recommendation

If either JupyterLab is not installed on the machine or a faster training is desired, [Google Colab](https://colab.research.google.com/) can be used. It does provide free GPU for a limited amount of time, but more than sufficient to train the network.

## References

[[1](https://www.cs.toronto.edu/~kriz/cifar.html)] Cifar10 dataset \
[[2](https://arxiv.org/abs/1911.11929)] Chien-Yao Wang and Hong-Yuan Mark Liao and I-Hau Yeh and Yueh-Hua Wu and Ping-Yang Chen and Jun-Wei Hsieh. "CSPNet: A New Backbone that can Enhance Learning Capability of CNN." 2019.
