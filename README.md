# TF2-EvoNorm

This repository implements candidates B0 and S0 defined in
[Evolving Normalization-Activation Layers](https://arxiv.org/pdf/2004.02967.pdf) in TensorFlow 2.0.

## Benchmark

As of today, the layers have been tested in a ResNet18 architecture against CIFAR10 & 100. It shows some gain 
performance compared to a BN-ReLU ResNet18.

All TensorBoards logs are available [here](https://tensorboard.dev/experiment/QmwLVEBvSd2k9pN1AjTZsg/#scalars).

## Requirements

![Tensorflow Versions](https://img.shields.io/badge/TensorFlow-2.x-blue)

If you want to launch the training, you will need to install [Click](https://click.palletsprojects.com/en/7.x/) and
[Loguru](https://github.com/Delgan/loguru).

## EvoNorm S0

<img src="./assets/evonorm-s0.png" width="500" height="300" />

## EvoNorm B0

<img src="./assets/evonorm-b0.png" width="500" height="300" />
