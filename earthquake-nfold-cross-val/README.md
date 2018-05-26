# earthquake-nfold-cross-val

This python program uses N-fold Cross Validation on multiple hyperparameter permutations. Within defined bounds, the program will generate multiple "hypotheses" (permutations of number of epochs, neural network hidden layer dimension, and optimizer learning rate) and then perform cross validation on each hypothesis.

**There are three input arguments:** 
* *'-M'* followed by an integer determines the number of hypotheses to generate
* *'-N'* followed by an integer determines how many "folds" to do in the cross validation
* *'-S'* followed by a float between 0 and 1 determines what proportion of the dataset to use as the test set
* *'-p'* followed by a boolean determines whether to preprocess on raw data (if false will assume preprocessed data is already available)

Data used for this program can be retrieved from here: https://www.kaggle.com/usgs/earthquake-database
