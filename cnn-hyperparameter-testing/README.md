# cnn-hyperparameter-testing

This python program is just practice in making a basic convolutional neural network for classification. It is written such that there are two convolution layers, each with pooling, and also a fully connected layer to softmax for classification.

The data set used here is MNIST-fashion, which consists 28x28 pixel images of clothing items. Each image is classified as an integer 0-9 based on the class of clothing item it is. The dataset can be found here: https://www.kaggle.com/zalando-research/fashionmnist

When run, the program iterates through various combinations of hyperparameters (i.e. 72 permutations of learning rate, batch size, and number of nodes in fully-connected layer).

Here are results from running by varying the learning rate with values [0.1, 0.05, 0.01, 0.005, 0.001, 0.0001], the fully connected layer with sizes [8, 16, 32, 64, 128, 256], and batch sizes [64, 256]. This run took approximately 9 hours 40 minutes with an Intel i7-3630QM and NVIDIA GeForce GTX 670MX. Results were achieved with 25 epochs, where each epoch included sequentially running through the training data via the batch size.

![Alt text](results/test-acc_batch-64.png?raw=true "Test Accuracy on batch size 64")
![Alt text](results/test-acc_batch-256.png?raw=true "Test Accuracy on batch size 256")

Results can also be accessed in the `.csv` files in the `results` folder. `test_results-epochs_25.csv` shows the cross-entropy cost and accuracy on the test data for each permutation. All of the rest of the files show the training cost and accuracy at the end of each epoch for every permutation.
