# cnn-hyperparameter-testing

This python program is just practice in making a basic convolutional neural network for classification. It is written such that there are two convolution layers, each with pooling, and also a fully connected layer to softmax for classification.

The data set used here is MNIST-fashion, which consists 28x28 pixel images of clothing items. Each image is classified as an integer 0-9 based on the class of clothing item it is. The dataset can be found here: https://www.kaggle.com/zalando-research/fashionmnist

When run, the program iterates through various combinations of hyperparameters (i.e. 72 permutations of learning rate, batch size, and number of nodes in fully-connected layer.
