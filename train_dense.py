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
import numpy as np
import os
import tensorflow as tf

def modify(X, Y):
    X_mod = np.ndarray((X.shape[0], 28*28))
    for i in range(X.shape[0]):
        X_mod[i, :, None] = np.reshape(X[i, :, :], (28*28, 1))
    
    Y_mod = np.zeros((X.shape[0], 10))
    for i in range(Y.shape[0]):
        Y_mod[i, Y[i]] = 1
    print(X_mod.shape)
    print(Y_mod.shape)
    
    return X_mod, Y_mod

def modifyX(X):
    X_mod = np.ndarray((X.shape[0], 28*28))
    for i in range(X.shape[0]):
        X_mod[i, :, None] = np.reshape(X[i, :, :], (28*28, 1))
    
    
    return X_mod

def train(trainX, trainY):
    '''
    Complete this function.
    '''
    
    trainX_mod, trainY_mod = modify(trainX, trainY)
    testX_mod, testY_mod = modify(testX, testY)
    learning_rate = 0.001
    training_epochs = 5
    batch_size = 100

    input_dim = 28*28

    n_input = input_dim
    n_hidden_1 = int(input_dim/2) + 5
    n_output = 10

    x = tf.placeholder('float', [None, n_input])
    y = tf.placeholder('float', [None, n_output])

    def mperceptron(x, weights, biases):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)

        # layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        # layer_2 = tf.nn.relu(layer_2)

        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        return out_layer

    weights = {
        'h1' : tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.001)),
        # 'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.001)),
        'out' : tf.Variable(tf.random_normal([n_hidden_1, n_output], 0, 0.001))
    }

    biases = {
        'b1' : tf.Variable(tf.random_normal([n_hidden_1], 0, 0.001)),
        # 'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.001)),
        'out' : tf.Variable(tf.random_normal([n_output], 0, 0.001))
    }

    pred = mperceptron(x, weights, biases)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    sess = tf.Session()    
    sess.run(init)
    
    epochs = 5
    batches = 100
    batch_size = int((trainX_mod.shape[0])/batches)

    for _ in range(epochs):
        for bx in range(batches):
            batch_indices = list(range((bx*batch_size), (bx + 1)*batch_size))
            _, c = sess.run([optimizer, cost], \
                            feed_dict={x: trainX_mod[batch_indices, :],\
                                       y: trainY_mod[batch_indices]})
    
    
    w_h1 = sess.run(weights)['h1']
    w_out = sess.run(weights)['out']
    b_h1 = sess.run(biases)['b1']
    b_out = sess.run(biases)['out']
    
    pickle.dump({
            'w_h1': w_h1,
            'w_out': w_out,
            'b_h1': b_h1,
            'b_out': b_out
        }, open('./saved_dense_matrices.p', 'wb'))
    
    sess.close()
    
def test(testX):
    '''
    Complete this function.
    This function must read the weight files and
    return the predicted labels.
    The returned object must be a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array should contain the label of the i-th test
    example.
    '''

    testX_mod = modifyX(testX)
    learning_rate = 0.001
    training_epochs = 5
    batch_size = 100

    input_dim = 28*28

    n_input = input_dim
    n_hidden_1 = int(input_dim/2) + 5
    n_output = 10

    x = tf.placeholder('float', [None, n_input])
    y = tf.placeholder('float', [None, n_output])

    def mperceptron(x, weights, biases):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)

        # layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        # layer_2 = tf.nn.relu(layer_2)

        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        return out_layer

    loader = pickle.load(open('./saved_dense_matrices.p', 'rb'))
    weights = {
        'h1' : tf.Variable(loader['w_h1']),
        # 'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.001)),
        'out' : tf.Variable(loader['w_out'])
    }

    biases = {
        'b1' : tf.Variable(loader['b_h1']),
        # 'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.001)),
        'out' : tf.Variable(loader['b_out'])
    }

    pred = mperceptron(x, weights, biases)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    sess = tf.Session()    
    sess.run(init)
    
    y_hat = sess.run(pred, feed_dict={x: testX_mod})
    
    sess.close()
    
    return np.argmax(y_hat, axis=1)
