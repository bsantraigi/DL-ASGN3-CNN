'''
Deep Learning Programming Assignment 2
--------------------------------------
Name: Bishal Santra
Roll No.: 12EC35001

======================================
Complete the functions in this file.
Note: Do not change the function signatures of the train
and test functions
'''
# Imports for CNN
import numpy as np
import os
import train_dense
import train_cnn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

# Imports for downloader

import requests
import os

def downloadData_CNN():
    http_proxy  = "http://10.3.100.207:8080"
    https_proxy = "http://10.3.100.207:8080"
    ftp_proxy   = "http://10.3.100.207:8080"

    proxyDict = { 
                  "http"  : http_proxy, 
                  "https" : https_proxy, 
                  "ftp"   : ftp_proxy
                }
    
    base = 'https://raw.githubusercontent.com/studiobytestorm/DL-ASGN3-CNN/master/tmp/'

    weightList = ['checkpoint', 'events.out.tfevents.1489824501.MSI', 'graph.pbtxt', 'model.ckpt-1.index',\
                  'model.ckpt-1.meta', 'model.ckpt-2000.data-00000-of-00001', 'model.ckpt-2000.index', 'model.ckpt-2000.meta']

    path_root = './dtmp/'
    if not os.path.exists(path_root):
        os.makedirs(path_root)
    
    count = 0
    total = len(weightList)
    
    print('Downloading CNN Weights: 1 of the files (12 MB in size) will take some time!')
    for name in weightList:
        url = base + name
        if os.path.isfile(path_root + name):
            print('File exists: [{}] -> Skip Download'.format(name))
            continue
        print('Downloading File: ', name)
        stream = requests.get(url, proxyDict)
        np_f = open(path_root + name, 'wb')
        np_f.write(stream.content)
        np_f.close()
        print(count+1,'/',total,' Complete')
        count += 1
        
    print('Download Complete')

    
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="SGD")

    # Generate Predictions
    predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
    }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def train(trainX, trainY):
    '''
    Complete this function.
    Note: Load model in folder ./tmp/
    '''
    
    train_data = np.asarray(trainX, dtype=np.float32) # Returns np.array
    train_labels = np.asarray(trainY, dtype = np.float32)

    # Estimator
    mnist_classifier = learn.Estimator(
      model_fn=cnn_model_fn, model_dir="./tmp")
    
    mnist_classifier.fit(x=train_data, y=train_labels, batch_size=100, steps=2000)


def test(testX):
    '''
    Complete this function.
    This function must read the weight files and
    return the predicted labels.
    The returned object must be a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array should contain the label of the i-th test
    example.
    
    
    Note: Download and Load model in folder ./dtmp/
    '''
    # Download the model
    downloadData_CNN()
    
    # Predict    
    eval_data = np.asarray(testX, dtype = np.float32) # Returns np.array

    # Estimator
    mnist_classifier = learn.Estimator(
      model_fn=cnn_model_fn, model_dir="./dtmp")
    
    yd_hat = list(mnist_classifier.predict(x=eval_data))
    labels_hat = np.array([what['classes'] for what in yd_hat])

    return labels_hat
