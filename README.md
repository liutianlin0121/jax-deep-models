# Deep learning models in JAX and Flax


Welcome to this repository of deep learning models written in [JAX](https://github.com/google/jax) and [Flax](https://flax.readthedocs.io/en/latest/)! JAX and Flax are neural network frameworks developed by Google that provide efficient, scalable, and flexible ways of building deep learning models.


This repository contains a collection of models organized by model types, such as multilayer perceptrons, convolutional neural networks, and autoencoders. The models are trained on various datasets, such as MNIST and CIFAR-10, and can be run in Jupyter notebooks or Google Colab.

This repository is inspired by [Sebastian Raschka](https://github.com/rasbt)'s [Deep Learning Model Zoo](https://sebastianraschka.com/deep-learning-resources/), which is written in PyTorch and Tensorflow; in addition, I have adapted code from various sources, including the [Flax examples](https://github.com/google/flax/tree/main/examples), the [UvA DL tutorials](https://uvadlc-notebooks.readthedocs.io/en/latest/), and the [JAXopt examples](https://jaxopt.github.io/stable/notebooks/index.html#deep-learning) :pray:.



## Multilayer Perceptron (MLP)

|Title | Dataset | Notebooks |
| --- | --- | --- |
| Basic MLP | MNIST  | [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/liutianlin0121/jax-deep-learning-models/blob/main/MLP/mlp-mnist.ipynb) &nbsp; &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liutianlin0121/jax-deep-learning-models/blob/main/MLP/mlp-mnist.ipynb)|


## Convolutional neural networks (ConvNets)

|Title | Dataset | Notebooks |
| --- | --- | --- |
| Basic ConvNet | MNIST  | [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/liutianlin0121/jax-deep-learning-models/blob/main/ConvNet/convnet-mnist.ipynb)  &nbsp; &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liutianlin0121/jax-deep-learning-models/blob/main/ConvNet/convnet-mnist.ipynb) |
| Basic ConvNet | CIFAR-10  | [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/liutianlin0121/jax-deep-learning-models/blob/main/ConvNet/convnet-cifar.ipynb)  &nbsp; &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liutianlin0121/jax-deep-learning-models/blob/main/ConvNet/convnet-cifar.ipynb) |
| Basic ConvNet with dropout| CIFAR-10  | [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/liutianlin0121/jax-deep-learning-models/blob/main/ConvNet/convnet-cifar-dropout.ipynb)  &nbsp; &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liutianlin0121/jax-deep-learning-models/blob/main/ConvNet/convnet-cifar-dropout.ipynb) |
| Basic ConvNet with batchnorm| CIFAR-10  | [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/liutianlin0121/jax-deep-learning-models/blob/main/ConvNet/convnet-cifar-batchnorm.ipynb)  &nbsp; &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liutianlin0121/jax-deep-learning-models/blob/main/ConvNet/convnet-cifar-batchnorm.ipynb) |
| ResNet | CIFAR-10  | [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/liutianlin0121/jax-deep-learning-models/blob/main/ConvNet/resnet-cifar.ipynb)  &nbsp; &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liutianlin0121/jax-deep-learning-models/blob/main/ConvNet/resnet-cifar.ipynb) |



## Autoencoders

|Title | Dataset | Notebooks |
| --- | --- | --- |
| MLP autoencoder | MNIST  | [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/liutianlin0121/jax-deep-learning-models/blob/main/AE/ae-mlp-mnist.ipynb)  &nbsp; &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liutianlin0121/jax-deep-learning-models/blob/main/AE/ae-mlp-mnist.ipynb) |
| Conv autoencoder | MNIST  | [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/liutianlin0121/jax-deep-learning-models/blob/main/AE/ae-convnet-mnist.ipynb)  &nbsp; &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liutianlin0121/jax-deep-learning-models/blob/main/AE/ae-convnet-mnist.ipynb) |
| Variational MLP autoencoder | MNIST  | [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/liutianlin0121/jax-deep-learning-models/blob/main/AE/vae-mlp-mnist.ipynb)  &nbsp; &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liutianlin0121/jax-deep-learning-models/blob/main/AE/vae-mlp-mnist.ipynb) |
| Variational Conv autoencoder | MNIST  | [![nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.org/github/liutianlin0121/jax-deep-learning-models/blob/main/AE/vae-conv-mnist.ipynb)  &nbsp; &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liutianlin0121/jax-deep-learning-models/blob/main/AE/vae-conv-mnist.ipynb) |





### Acknowledgement

This repository includes code that has been adapted from various sources, including the [Flax examples](https://github.com/google/flax/tree/main/examples), the [UvA DL tutorials](https://uvadlc-notebooks.readthedocs.io/en/latest/), and the [JAXopt examples](https://jaxopt.github.io/stable/notebooks/index.html#deep-learning).


### Disclaimer
All notebooks in this repository are written for didactic purposes and are not intended to serve as performance benchmarks.
