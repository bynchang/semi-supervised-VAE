# Semi-Supervised Learning with Variational Autoencoder
Implementation of Semi-Supervised Learning with Deep Generative Models (Kingma, 2014) (https://arxiv.org/abs/1406.5298).
\newline
The goal of semi-supervised learning is to train a model using both labeled data and unlabeled data. Using a deep generative model approach (VAE), we are able to learn a latent representation of the data and train a classifier at the same time. In this example, I train a convolutional variational autoencoder and use a convolutional neural network as my classifier. It successfully boost the baseline accuracy by 5% on the STL-10 dataset.
