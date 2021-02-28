# Replicating Glow and Investigating Its Performanceon OOD-detection

In this project we implement the flow-based deep generative model Glow from the paper "Glow: Generative Flow with Invertible 1x1 Convolutions" by Kingma and Dhariwal (https://arxiv.org/abs/1807.03039). We train the model on MNIST and reproduce some of the main results from the original paper. Additionally, we show that the trained
model can generate new, realistic images, and we find that linear interpolation in latent space produces realistic images with smooth transitions. Finally, we investigate
the model's performance on Out-of-Distribution (OOD) detection using typicality tests. The implementation is done in TensorFlow using Keras and the project was done as part of the course DD2412 Advanced Deep Learning at KTH.

