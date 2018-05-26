import argparse
import datetime   as dt
import math
import os

import tensorflow as tf
import numpy      as np
import pandas     as pd
from sklearn.cross_validation import train_test_split

from tqdm import tqdm

# Suppress TensorFlow console logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Used as seed for reproducibility
RAND_SEED = np.random.randint(low=0, high=100)

# File path to raw data file
RAW_DATA_PATH = './data_raw.csv'
PREPROC_DATA_PATH = 'data_preproc.csv'

# Defaults for input arguments for program
DEFAULT_NUM_HYPOTHESES = 10
DEFAULT_N_VALIDATION   = 10
DEFAULT_TEST_SIZE      = 0.15

# Bounds for random selection for each hypothesis
MIN_EPOCHS = 1000
MAX_EPOCHS = 2000
MIN_H_DIM  = 4
MAX_H_DIM  = 30
MIN_L_RATE = 0.00001
MAX_L_RATE = 0.1

# Data has three features in, and simply predicting one value
NN_INPUT_WIDTH = 3 # Only using 3 input features: Date, Latitude, Longitude
NN_OUT_WIDTH   = 1 # Predicting 1 value: Magnitude

# Generate "hypotheses, where each hypothesis is just a permutation of
# number of epochs, hidden-layer dimension, and learning rate
def generate_hypotheses(num_hypotheses):
    hypotheses = pd.DataFrame()
    hypotheses['num_epochs'] = \
        np.random.randint(low=MIN_EPOCHS,high=MAX_EPOCHS, size=num_hypotheses)
    hypotheses['h_dim'] = \
        np.random.randint(low=MIN_H_DIM,high=MAX_H_DIM, size=num_hypotheses)
    hypotheses['l_rate'] = \
        np.random.uniform(low=MIN_L_RATE, high=MAX_L_RATE, size=num_hypotheses)
    return hypotheses

# Retrieve data, optionally skip preprocess if it has already been done before
def get_data(do_preprocess):
    data = None
    try:
        if do_preprocess:
            data = pd.read_csv('database.csv')
            data = data[list(data.columns[:9])]
            data = data.drop(['Type','Time','Depth', 'Depth Error', 
                              'Depth Seismic Stations'], axis=1)
            data['Date'] = pd.to_datetime(data['Date'])

            # Do min_max_scaling
            min_date = data['Date'][0]
            max_date = data['Date'][len(data['Date'])-1]
            min_max_del_date = (data['Date'][len(data['Date'])-1] 
                                  - data['Date'][0]).days
            data['Date'] = [(data['Date'][i] - min_date).days 
                            / min_max_del_date
                            for i in range(len(data['Date']))]

            #
            #min_lat = np.min(data['Latitude'])
            #max_lat = np.max(data['Latitude'])
            #min_max_del_lat = max_lat - min_lat
            #data['Latitude'] = [(data['Latitude'][i] - min_lat) 
            #                    / min_max_del_lat
            #                    for i in range(len(data['Latitude']))]

            #min_lon = np.min(data['Longitude'])
            #max_lon = np.max(data['Longitude'])
            #min_max_del_lon = max_lon - min_lon
            #data['Longitude'] = [(data['Longitude'][i] - min_lon) 
            #                    / min_max_del_lon 
            #                    for i in range(len(data['Longitude']))]

            min_mag = np.min(data['Magnitude'])
            max_mag = np.max(data['Magnitude'])
            min_max_del_mag = max_mag - min_mag
            data['Magnitude'] = [(data['Magnitude'][i] - min_mag) 
                                / min_max_del_mag 
                                for i in range(len(data['Magnitude']))]
            
            # Restore the magnitude after prediction as a function to later
            # see average distance of predicted value from true value
            def restore_mag(scaled_mag):
                return scaled_mag*(min_max_del_mag) + min_mag
            restore_mag = np.vectorize(restore_mag)

            # Shuffle data
            data = data.sample(frac=1).reset_index(drop=True)

            # Save data to csv for future runs
            data.to_csv(PREPROC_DATA_PATH, index=False)

        # Else grab already preprocessed data
        else:
            data = pd.read_csv(PREPROC_DATA_PATH)
    except:
        print("ERROR: Issue while loading or processing data. \
               (preprocessing: %s"%(str(do_preprocess)))

    return restore_mag, data


def train_and_test_model(data_train, data_test, epochs, h_dim, lr, restore_mag):
    
    # Build model #################################################
    x = tf.placeholder(tf.float32, shape=(None,NN_INPUT_WIDTH))
    y = tf.placeholder(tf.float32, shape=(None,NN_OUT_WIDTH))

    W1 = tf.Variable(np.random.rand(NN_INPUT_WIDTH,h_dim),dtype=tf.float32)
    b1 = tf.Variable(np.zeros((1,h_dim)), dtype=tf.float32)
    W2 = tf.Variable(np.random.rand(h_dim,NN_OUT_WIDTH), dtype=tf.float32)
    b2 = tf.Variable(np.zeros((1,NN_OUT_WIDTH)), dtype=tf.float32)

    h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
    pred = tf.matmul(h, W2) + b2

    cost = tf.reduce_sum(tf.pow(y - pred, 2)
                         /(tf.cast(tf.shape(x)[0], tf.float32)))
    optimizer = tf.train.AdagradOptimizer(lr).minimize(cost)
    init = tf.global_variables_initializer()
    ###############################################################

    with tf.Session() as sess:
        sess.run(init)

        in_x = data_train.loc[:,['Date','Latitude','Longitude']].as_matrix()
        in_y = data_train.loc[:,['Magnitude']].as_matrix()

        # Train model with chosen train data
        for e in tqdm(range(epochs)):
            sess.run(optimizer, feed_dict={x:in_x, y:in_y})
        train_error = sess.run(cost,feed_dict={x: in_x, y:in_y})
        

        # Optional: print some true/predicted values from train set
        
        # train_pred = restore_mag(sess.run(pred,feed_dict={x: in_x, y:in_y}))
        # train_true = restore_mag(in_y)
        # avg_diff_train = np.average(np.absolute(train_pred - train_true))
        # print("TrainPred:",train_pred[0:10])
        # print("TrainActual:",train_true[0:10])

        # Check error for validation set
        in_x = data_test.loc[:,['Date','Latitude','Longitude']].as_matrix()
        in_y = data_test.loc[:,['Magnitude']].as_matrix()
        test_error = sess.run(cost,feed_dict={x: in_x, y:in_y})
        
        # Optional: print some true/predicted values from test set
        
        # test_pred = restore_mag(sess.run(pred,feed_dict={x: in_x, y:in_y}))
        # test_true = restore_mag(in_y)
        # avg_diff_test = np.average(np.absolute(test_pred - test_true))
        # print("TestPred:",test_pred[0:10])
        # print("TestActual:",test_true[0:10])
        # print("\t\tDebugging: avg_diff_train: %.6f, avg_diff_test: %.6f"%\
        #    (avg_diff_train, avg_diff_test))

    return train_error, test_error

# Train and validate on dataset using n-fold cross-validation
def cross_validate(n, data, epochs, h_dim, lr, restore_mag):
    
    val_errors = list()
    # Get size of each validation set based on number of folder
    val_size = math.ceil(len(data.index)/n)
    
    # Perform n-fold cross validation
    for i in range(n):
        print("\tStarting train and validate on fold (%d/%d)"%(i+1,n))
        
        # Select portions of data set to use for training and validation
        if i == n-1:
            # If last fold, the make sure to use the last val_size values
            # of data set since dataset size may not be evenly divisible by n
            data_val = data.iloc[(len(data.index)-val_size):, :]
            data_train = data.iloc[:(len(data.index)-val_size), :]
        else:
            data_val = data.iloc[(i*val_size):((i+1)*val_size)]
            data_train = data.iloc[0:(i*val_size), :].append( \
                data.iloc[((i+1)*val_size):, :])

        # Train model and get validation error for hypothesis
        _, val_error = train_and_test_model(data_train=data_train, 
                                            data_test=data_val, 
                                            epochs=epochs, 
                                            h_dim=h_dim, 
                                            lr=lr,
                                            restore_mag=restore_mag)
        val_errors.append(val_error)

    return float(np.average(val_errors))

###############################################################################
###############################################################################
###############################################################################

#############
# MAIN CODE #
#############
def run(num_hypotheses, n_fold, test_size, do_preprocess):
    
    hypotheses = generate_hypotheses(num_hypotheses)

    restore_mag, data = get_data(do_preprocess)
    data_train, data_test = train_test_split(data, test_size=test_size,
                                             random_state = RAND_SEED)
    
    # Run N-fold cross validation on all hypotheses
    hyp_errors = list()
    for i in range(num_hypotheses):
        print("Starting hypothesis (%d/%d)"%(i+1,num_hypotheses))
        hyp_error = cross_validate(n=n_fold, data=data_train,
                                       epochs=hypotheses['num_epochs'][i],
                                       h_dim=hypotheses['h_dim'][i],
                                       lr=hypotheses['l_rate'][i],
                                       restore_mag=restore_mag)
        hyp_errors.append(hyp_error)

    
    # Print results from N-fold cross validation
    for i in range(num_hypotheses):
        print("H_%d (num_epochs:%d, h_dim:%d, l_rate:%.6f) CV_ERROR: %.7f"%\
            (i+1,hypotheses['num_epochs'][i],hypotheses['h_dim'][i],
             hypotheses['l_rate'][i],hyp_errors[i]))

    # Train on full train set and get error using test set for all hypotheses
    print("\nNow doing full train on hypotheses and running on test data...\n")
    for i in range(num_hypotheses):
        train_error, test_error = train_and_test_model(data_train=data_train,
                                         data_test=data_test,
                                         epochs=hypotheses['num_epochs'][i],
                                         h_dim=hypotheses['h_dim'][i],
                                         lr=hypotheses['l_rate'][i],
                                         restore_mag=restore_mag)
        print("H%d, train error: %f, test error: %f"%(i+1,train_error,
                                                     test_error))




def str2bool(input):
    if input.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif input.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

#########################
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Perform N-fold cross\
                                     validation on several hypotheses.")
    parser.add_argument('-M', type=int, default=DEFAULT_NUM_HYPOTHESES)
    parser.add_argument('-N', type=int, default=DEFAULT_N_VALIDATION)
    parser.add_argument('-S', type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument('-p', nargs='?', type=str2bool, 
                        const=True, default=True)
    
    np.random.seed(RAND_SEED)
    args = parser.parse_args()

    # Run main code
    run(num_hypotheses=args.M, n_fold=args.N, 
        test_size=args.S, do_preprocess=args.p)              
#########################
