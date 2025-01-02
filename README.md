# Gradient Routing Implementation

This repository contains an implementation of the paper ["Gradient Routing: Masking Gradients to Localize Computation in Neural Networks"](https://arxiv.org/abs/2410.04332). The code specifically replicates the MNIST autoencoder experiments described in Section 4.1 of the paper. Some training details differ, but this should give a solid start for building on the experiments in the paper.

<b>Repository Status:</b> The respository is currently being updated with more experiments and replications. The plan it to eventually use this respository as starter code for research projects related to gradient routing for UChicago's AI Safety club. Pull requests are welcome!

## Overview

Gradient routing is a training method that isolates capabilities to specific regions of a neural network by applying data-dependent masks to stop gradients in certain regions of the network during backpropagation. This implementation demonstrates how gradient routing can split MNIST digit representations into distinct halves of an autoencoder's latent space.

If we train a encoder/decoder architechure on the MNIST handwritten digit dataset, we can route gradients to localize representations of digits 0-4 in the bottom half of the latent space and digits 5-9 in the top. We can confirm that this works by training decoders with only access to one half of the latent space. The image below shows original images (from the validation set) on the top and images generated from a decoder with only access to the top half of the latent space. As you can see, performance is reasonable for these digits. 

![top_cert_high_digits](https://github.com/user-attachments/assets/7a8fc27c-6912-477b-a057-da8a2c8246bc)

However, when you test this decoder on digits 0-4 (from the validation set) the performance is much worse. This is because we routed gradients that learn from these digits away from the top half of the latent space:

![top_cert_low_digits](https://github.com/user-attachments/assets/e222e78d-29a5-4802-8140-c77a1cd42224)

You can see a similar effect when you train a decoder on only the bottom half of the encoding:

![bottom_cert_low_digits](https://github.com/user-attachments/assets/e09a3eb6-b63c-41aa-abb1-256263bf14aa)

![bottom_cert_high_digits](https://github.com/user-attachments/assets/f59c6c77-11c1-4b2e-8924-acec1bf2d958)

Overall, this shows that we have sucessfully isolated representations of certain features to one side of the latent space with representations of other features to the other. This can be shown more robustly through comparing measurements of MAE losses for both the top and bottom decoder on each digit in the validation set:

![compare_certifiers](https://github.com/user-attachments/assets/7ab731ab-f624-4b68-82f3-5abc0b6eae36)

## Key Features

- Implementation of gradient routing for MNIST autoencoder
- Replication of the architecture described in Section 4.1
- Visualization for analyzing digit reconstructions
