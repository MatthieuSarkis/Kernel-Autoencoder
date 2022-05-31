# Kernel Autoencoder

## Requirements

* python 3.8.10
* numpy
* scipy
* torch
* torchvision
* tqdm
* ipykernel

```shell
pip install -e .
```

## Preparing the MNIST and Fashion MNIST dataset

```shell
python src/data_factory/mnist.py --dataset mnist
```
or
```shell
python src/data_factory/mnist.py --dataset fashion_mnist
```

## Train a model

```shell
python src/autoencoder/main.py --dataset mnist
```
or
```shell
python src/autoencoder/main.py --dataset fashion_mnist
```

## License
[Apache License 2.0](https://github.com/MatthieuSarkis/Kernel-Autoencoder/blob/master/LICENSE)