# Gradient Routing Replication

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

<p align="center">
  <img width="600" src="https://github.com/user-attachments/assets/7ab731ab-f624-4b68-82f3-5abc0b6eae36">
</p>

## Training

The process to train the encoder/decoders should be almost the same as outlined in the original paper. Correlation loss is described in the appendix of the paper but is not included in the repository. Note that I also measure an L1 loss for the output of the decoder (as described in the original paper). In the paper, the authors train for 200 epochs while I trained for 400 epochs to get the results shown above. I will note that returns after epoch 200 are minimal.

<p align="center">
  <img width="600" src="https://github.com/user-attachments/assets/e8647fd8-a522-463e-a006-788f45362fba">
</p>

(Note that the purple line in image above represents the decoder loss for the bottom decoder. The reason that there is no line for the top decoder is because their losses are roughly the same and the lines would overlap.)

## Theory of Change

There are two main reasons why I find this research direction personally interesting:

1. Gradient routing can make models more interpretable. When we localize computation related to certain features, we will know where to find them later we want to understand more about a model's internals.
2. Related to #1, there are early results indicating that when we localize computation for one subject, the model routes realted concepts to the same area. This indicates that gradient routing is scalable to domains where there is limited labeled data.
3. It is a glimpse into a world where we have modules in models that we can turn on and off. If we are concerned that an area of a model is related to dangerous bahavior, we could shut it off.
4. 

## Acknowledgments

If you use any of the ideas from this repository in your own work please cite the original paper:

```
@article{cloud2024gradient,
	title={Gradient Routing: Masking Gradients to Localize Computation in Neural Networks},
	url={https://arxiv.org/abs/2410.04332v1},
	journal={arXiv.org},
	author={Cloud, Alex and Goldman-Wetzler, Jacob and Wybitul, Evžen and Miller, Joseph and Turner, Alexander Matt},
	year={2024},
}
```

Made with ❤️ and PyTorch
