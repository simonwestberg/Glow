# Replicating Glow and Investigating Its Performance on OOD-detection

In this project we implement the flow-based deep generative model Glow from the paper "Glow: Generative Flow with Invertible 1x1 Convolutions" by Kingma and Dhariwal (https://arxiv.org/abs/1807.03039). We train the model on MNIST and reproduce some of the main results from the original paper. Additionally, we investigate
the model's performance on Out-of-Distribution (OOD) detection using typicality tests. The implementation was done in TensorFlow using Keras and the project was done as part of the course DD2412 Advanced Deep Learning at KTH. The report detailing our method and findings can be seen [here](report.pdf)

## Architecture

The architecture of Glow builds heavily upon the work done in [NICE](https://arxiv.org/abs/1410.8516) and [Real NVP](https://arxiv.org/abs/1605.08803). The main contribution of the Glow paper was the introduction of an invertible 1x1 convolution in the flow for permuting the channel dimensions. Each step of flow in Glow consists of an activation normalization layer, an invertible 1x1 convolution, and an affine coupling layer. The architecture can be seen in the image below and is described in more detail in the [report](report.pdf).

![](/figures/glow.PNG)

*A single step of flow and the multi-scale architecture of Glow. Image from [here](https://arxiv.org/abs/1807.03039)*

&nbsp;

## Results 

When training the model on MNIST, we found that the 1x1 convolution achieved a lower average negative log-likelihood than both a shuffle and reversing operation, 
demonstrating the power of the learnable 1x1 convolutional permutation, a key result by Kingma and Dhariwal (see figure below).

![](/figures/compare_test_NLL.png)

*Average negative log-likelihood in bits/dimension of the MNIST test set. The 1x1 convolution clearly outperforms the shuffle and reverse operations.*

New samples generated by the model can be seen below. We also performed linear interpolation in latent space between two test images for each class (see below) and found that this 
produces realistic images with smooth transitions.

&nbsp;

![](/figures/samples.png)

*New samples generated by the model with temperature=0.7*

&nbsp;

![](/figures/interpolation.png)

*Linear interpolation in latent space*

&nbsp;



