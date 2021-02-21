import numpy as np
import tensorflow as tf
from function import *
import json


def compute_cost(S, Y, m):    
    cost = tf.reduce_sum(tf.pow(tf.transpose(S) - tf.transpose(Y), 2))
    sum_cost = tf.cast(cost, tf.float32) / (2 * m)
    return sum_cost

def back_propagation(_logger, X_train, X_test, T_train, T_test, bp_hparameters):
    """Back propagation based on the architecture PLN has constructed"""
    train_accuracy_lists = []
    train_NME_lists = []
    test_accuracy_lists = []
    test_NME_lists = []
    m = X_train.shape[1]
    P = X_train.shape[0]
    Q= T_train.shape[0]
    data = bp_hparameters["data"]
    learning_rate = bp_hparameters["learning_rate"]
    iteration_num = bp_hparameters["iteration_num"]

    _logger.info("Read parameters by SSFN")
    parameters_path = "./parameters/"
    filename = parameters_path + str(data) + '_n_lists.json'
    with open(file=filename, encoding='utf-8', mode="r") as f:
        n_lists = json.load(f)
        n_lists = [int(i) for i in n_lists]
    n_lists_length = len(n_lists)
    
    outputs = {}
    for i in range(1, n_lists_length +1):
        outputs["W" + str(i)]= np.load(parameters_path + data + '_W'+str(i) + '.npy').astype(np.float32)
        outputs["O" + str(i)]= np.load(parameters_path + data + '_O'+str(i) + '.npy').astype(np.float32)

    _logger.info("Construct the network in tensorflow")
    tf.reset_default_graph()
    X, T = create_placeholder(P, Q)
    if n_lists_length == 0:
        O = tf.get_variable(initializer=tf.constant(value=np.load(parameters_path + data + '_Wl' + '.npy')), \
                            name="O0", dtype=tf.float32)
        S = tf.matmul(tf.cast(O, tf.float32), X)
    else:
        Y = X
        for i in range(1, n_lists_length + 1):
            W = tf.get_variable(initializer=tf.constant(outputs["W"+str(i)]), name="W"+str(i), dtype=tf.float32)
            tmp_Z = tf.matmul(W, Y)
            Z_part1, Z_part2 = tmp_Z[:2*Q, :], tmp_Z[2*Q:, :]
            Z_part2 = Z_part2 / tf.reduce_sum(Z_part2**2, axis=0, keepdims=True)**(1/2)
            Z = tf.concat([Z_part1, Z_part2], axis=0)
            Y = tf.nn.relu(Z)

        O = tf.get_variable(initializer=tf.constant(value=outputs["O" + str(n_lists_length)]),\
                            name="O" +str(i), dtype=tf.float32)
        S = tf.matmul(tf.cast(O, tf.float32), Y)        
    cost = compute_cost(S, T, m)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        _logger.info("Start Optimization of back propagation")
        sess.run(init)
        for epoch in range(0, iteration_num+1):
            # predict for train
            S_value = sess.run(S, feed_dict={X: X_train})
            train_accuracy = calculate_accuracy(S_value, T_train)
            train_NME = compute_nme(S_value, T_train, m=X_train.shape[1])

            # predict for test
            S_test_value = sess.run(S, feed_dict={X: X_test})
            test_accuracy = calculate_accuracy(S_test_value, T_test)
            test_NME = compute_nme(S_test_value, T_test, m=X_test.shape[1])

            # preserve performance
            train_accuracy_lists.append(train_accuracy)
            train_NME_lists.append(train_NME)
            test_accuracy_lists.append(test_accuracy)
            test_NME_lists.append(test_NME)
            
            if epoch % 100 == 0:
                #_logger.info("Train Accuracy: {}".format(train_accuracy))
                #_logger.info("Train NME: {}".format(train_NME))
                _logger.info("Test Accuracy: {}".format(test_accuracy))
                #_logger.info("Test NME: {}".format(test_NME))
                
            # optimization
            _ = sess.run([optimizer], feed_dict={X:X_train, T:T_train})
    
    #_logger.info("Plot performance as the number of iteration increases")
 
    #plot_data_backprop(iteration_num, train_accuracy_lists, test_accuracy_lists, train_NME_lists, test_NME_lists, data, learning_rate)