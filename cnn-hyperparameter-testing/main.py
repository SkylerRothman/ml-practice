import numpy as np
import pandas as pd
import tensorflow as tf

import time
import math
from tqdm import tqdm

# Suppress TensorFlow console logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Random state seed for reproducibility
RAND_SEED = 55

FILEPATH_TRAIN = './fashion-mnist_train.csv'
FILEPATH_TEST = './fashion-mnist_test.csv'

IMG_SIZE_FLAT = 784
IMG_SIZE_SIDE = 28
NUM_CHANNELS = 1
NUM_CLASSES = 10
FILTER_SIZE_1 = 5
FILTER_NUM_1 = 16
FILTER_SIZE_2 = 5
FILTER_NUM_2 = 36

#BATCH_SIZE = 256
NUM_EPOCHS = 25
#LEARNING_RATE = 0.01

EPOCH_DISPLAY_STEP = 1

def get_data(train_data_filepath, test_data_filepath):
    data_train = pd.read_csv(train_data_filepath)
    data_test = pd.read_csv(test_data_filepath)

    # data['label'] has integer value 0 - 9
    # each value corresponds to a different clothing item

    return data_train, data_test


def convolution_layer(input, num_channels, filter_size, 
                          num_filters, use_pooling):
    conv_shape = (filter_size, filter_size, num_channels, num_filters)

    weights = tf.Variable(tf.truncated_normal(shape=conv_shape, 
                                              stddev=0.10, seed=RAND_SEED))
    biases = tf.Variable(tf.Variable(tf.constant(0.0, shape=[num_filters])))

    conv_layer = tf.nn.conv2d(input=input, filter=weights, 
                              strides=[1,1,1,1], padding='SAME') \
                 + biases

    if use_pooling:
        pool_layer=tf.nn.avg_pool(value=conv_layer, ksize=(1,2,2,1), 
                             strides=(1,2,2,1), padding='SAME')
        outer_layer = tf.nn.relu(pool_layer)
    else:
        outer_layer = tf.nn.relu(conv_layer)

    

    return outer_layer, weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, shape=(-1, num_features))
    return layer_flat, num_features

def fc_layer(input, num_inputs, num_outputs, use_relu):
    weights = tf.Variable(tf.truncated_normal(shape=(num_inputs, num_outputs),
                                              stddev=0.10, seed=RAND_SEED))
    biases = tf.Variable(tf.Variable(tf.constant(0.0, shape=[num_outputs])))

    layer = tf.matmul(input, weights) + biases

    if use_relu:
        outer_layer = tf.nn.relu(layer)
    else:
        outer_layer = layer

    return outer_layer

    

def make_model(fc_size, lr):
    x = tf.placeholder(dtype=tf.float32, shape=[None, IMG_SIZE_FLAT])

    #image shape: [BATCH_SIZE x IMG_SIZE_SIDE x IMG_SIZE_SIDE x NUM_CHANNELS]
    x_image = tf.reshape(x, shape=(-1, IMG_SIZE_SIDE, 
                                   IMG_SIZE_SIDE, NUM_CHANNELS))

    
    # output: [BATCH_SIZE x IMG_SIZE_SIDE/2 x IMG_SIZE_SIDE/2 x NUM_FILTERS_1]
    layer_conv_1, weights_conv_1 = convolution_layer(input=x_image, 
                                        num_channels=NUM_CHANNELS,
                                        filter_size=FILTER_SIZE_1,
                                        num_filters=FILTER_NUM_1,
                                        use_pooling=True)
 
    # output: [BATCH_SIZE x IMG_SIZE_SIDE/4 x IMG_SIZE_SIDE/4 x NUM_FILTERS_2]
    layer_conv_2, weights_conv_2 = convolution_layer(input=layer_conv_1, 
                                        num_channels=FILTER_NUM_1,
                                        filter_size=FILTER_SIZE_1,
                                        num_filters=FILTER_NUM_2,
                                        use_pooling=True)

    # output is [BATCH_SIZE x ((IMG_SIZE_SIDE/4)^2 *NUM_FILTERS_2)]
    layer_flat, num_features = flatten_layer(layer_conv_2)

    # output is [BATCH_SIZE x fc_size]
    layer_fc_1 = fc_layer(input=layer_flat, num_inputs=num_features,
                          num_outputs=fc_size, use_relu=True)
    # output is [BATCH_SIZE x NUM_CLASSES]
    layer_fc_2 = fc_layer(input=layer_fc_1, num_inputs=fc_size,
                          num_outputs=NUM_CLASSES, use_relu=False)
    
    # Get most likely class for each input image in batch
    y_pred = tf.nn.softmax(logits=layer_fc_2, axis=1)
    y_pred_class = tf.argmax(y_pred, axis=1)

    y_true = tf.placeholder(dtype=tf.int64, shape=(None, NUM_CLASSES),
                            name='y_true')
    y_true_class = tf.argmax(y_true, axis=1)
    
    # Calculate loss and make optimizer
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(\
                                logits=layer_fc_2, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)\
                         .minimize(cost)
    
    # Get accuracy
    correct_prediction = tf.equal(y_pred_class, y_true_class)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return x, y_true, cost ,optimizer, accuracy


def train_and_test(data_train, data_test, num_epochs, batch_size, fc_size, lr):

    x_train = data_train.drop('label', axis=1).as_matrix()
    # convert y to one_shot class labels
    y_train = (pd.get_dummies((data_train['label']))).as_matrix()
    
    x, y_true, cost ,optimizer, accuracy= make_model(fc_size, lr)

    # data that will be returned
    test_acc = 0.0  # Will be assigned
    test_cost = 0.0 # Will be assigned
    train_info = pd.DataFrame(columns=['epochs', 
                                       'train_accuracy', 
                                       'train_cost'])
    

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # Track training progress by epoch index
        info_index = 0

        ###################################################################
        # Train with number of epochs and train in batches for each epoch #
        ###################################################################
        num_batches = math.ceil(y_train.shape[0]/batch_size)
        for epoch in tqdm(range(num_epochs)):
            # Each epoch iterates through training data sequentially in batches
            for batch_num in range(num_batches):
                # Determine indices for batch in training data
                batch_start_index = batch_size*batch_num
                batch_end_index = min(batch_size*(batch_num+1),
                                      y_train.shape[0])
                x_train_batch = x_train[batch_start_index:batch_end_index]
                y_train_batch = y_train[batch_start_index:batch_end_index]


                # Optimize function for batch
                feed_dict_train = {x: x_train_batch, y_true: y_train_batch}
                session.run(optimizer, feed_dict=feed_dict_train)
            
            ############################
            # Record training progress #
            ############################
            if (epoch % EPOCH_DISPLAY_STEP == 0):
                train_acc = list()
                train_cost = list()
                for batch_num in range(num_batches):
                    # Determine indices for batch in training data
                    batch_start_index = batch_size*batch_num
                    batch_end_index = min(batch_size*(batch_num+1),
                                          y_train.shape[0])
                    x_train_batch = x_train[batch_start_index:batch_end_index]
                    y_train_batch = y_train[batch_start_index:batch_end_index]
                    
                    feed_dict_train = {x: x_train_batch, y_true: y_train_batch}
                    train_acc.append(float(session.run(\
                                          accuracy,feed_dict=feed_dict_train)))
                    train_cost.append(float(session.run(\
                                          cost,feed_dict=feed_dict_train)))
                
                # Store info
                train_info.loc[info_index] = [epoch, 
                                             np.average(train_acc),
                                             np.average(train_cost)]
                info_index = info_index + 1
            # END TRAINING RECORDING ###
        # END TRAINING ####################################################

        #########################
        # Test on trained model #
        #########################
        x_test = data_test.drop('label', axis=1).as_matrix()
        # convert y to one_shot class labels
        y_test = (pd.get_dummies((data_test['label']))).as_matrix()
            
        feed_dict_test = {x: x_test, y_true: y_test}
        test_acc = session.run(accuracy, feed_dict=feed_dict_test)
        test_cost = session.run(cost, feed_dict=feed_dict_test)
        # END TEST ##############

    return train_info, test_acc, test_cost
            
###############################################################################

# main code
def run():
    data_train, data_test = get_data(FILEPATH_TRAIN, FILEPATH_TEST)

    fc_sizes = [8, 16, 32, 64, 128, 256]
    learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]
    batch_sizes = [64, 256]

    test_info = pd.DataFrame(columns=['fc_size', 
                                       'learning_rate', 
                                       'batch_size',
                                       'test_cost',
                                       'test_accuracy'])
    info_index = 0 

    for fc_size in fc_sizes:
        for learning_rate in learning_rates:
            for batch_size in batch_sizes:
                train_info, test_acc, test_cost = train_and_test(\
                                                         data_train=data_train,
                                                         data_test=data_test,
                                                         num_epochs=NUM_EPOCHS,
                                                         batch_size=batch_size,
                                                         fc_size=fc_size,
                                                         lr=learning_rate)
                print("--------------------------")
                print("FC Layer Size: ", fc_size)
                print("Learning Rate: ", learning_rate)
                print("Batch Size: ", batch_size)
                print("Test Accuracy: ", test_acc)
                print("Test Cost: ", test_cost)
                print("--------------------------")

                test_info.loc[info_index] = [fc_size, learning_rate,
                                             batch_size, test_cost,
                                             test_acc]
                info_index = info_index + 1

                test_info.to_csv('./results/test_results-epochs_%d.csv'\
                                    %(NUM_EPOCHS))
                train_info.to_csv('./results/fc_%d-lr_%.3f-bs_%d-eps_%d.csv'\
                                    %(fc_size, learning_rate, 
                                      batch_size, NUM_EPOCHS))

    




#########################
if __name__ == '__main__': 
    run();
#########################



