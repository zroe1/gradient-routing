# Gradient Routing Implementation

This repository contains an implementation of the paper ["Gradient Routing: Masking Gradients to Localize Computation in Neural Networks"](https://arxiv.org/abs/2410.04332). The code specifically replicates the MNIST autoencoder experiments described in Section 4.1 of the paper. Some training details differ, but this should give a solid start for building on the experiments in the paper.

The respository is currently being updated with more experiments and replications. Pull requests are welcome!

## Overview

Gradient routing is a training method that isolates capabilities to specific regions of a neural network by applying data-dependent masks to stop gradients in certain regions of the network during backpropagation. This implementation demonstrates how gradient routing can split MNIST digit representations into distinct halves of an autoencoder's latent space.

## Key Features

- Implementation of gradient routing for MNIST autoencoder
- Replication of the architecture described in Section 4.1
- Visualization for analyzing digit reconstructions
