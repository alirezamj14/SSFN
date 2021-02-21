# -*- coding: utf-8 -*-

import numpy as np
import logging
from function import *
import math


def optimize_each_layer(_logger, X_train, X_test, T_train, T_test, outputs, n_lists, lth_layer, ssfn_hparameters):
    """Build the structure on l'th layer"""
    P = X_train.shape[0]
    Q = T_train.shape[0]
    alpha = ssfn_hparameters["alpha"]
    eta_n = ssfn_hparameters["eta_n"]
    max_n = ssfn_hparameters["max_n"]
    max_k = ssfn_hparameters["max_k"]
    delta = ssfn_hparameters["delta"]
    myu = ssfn_hparameters["myu"]
    
    # accuracy and NME on each layer
    el_train_accuracy_lists = []
    el_test_accuracy_lists = []
    el_train_NME_lists = []
    el_test_NME_lists = []
    
    # set the initialized parameter
    nodes = 2 * Q
    prev_nodes = n_lists[lth_layer-2] if lth_layer != 1 else P
    cache_n = {}
    V = create_v_values(Q)
    
    # set up  parameter on previous layer
    prev_O = outputs["O"+str(lth_layer-1)] if lth_layer != 1 else outputs["Wl"]
    prev_Y = outputs["Y" + str(lth_layer -1)] if lth_layer != 1 else X_train
    prev_Y_test = outputs["Y_test" + str(lth_layer -1)] if lth_layer != 1 else X_test
    
    max_iter_num = int(max_n/delta) + 1
    for iter_j in range(max_iter_num):
        nodes = 2 * Q + iter_j * delta
        # set the R and W values
        if iter_j == 0:
            R = np.array([], dtype=np.float32)
            W = np.dot(V, prev_O)
        else:
            R = 2 * np.random.rand(nodes- 2 * Q, prev_nodes) - 1 if iter_j == 1 else np.concatenate([R, 2 * np.random.rand(delta, prev_nodes) - 1], axis=0)
            W = np.concatenate([np.dot(V, prev_O), R], axis=0)
        tmp_Z = np.dot(W, prev_Y)
        Z = normalize_Z(tmp_Z, Q)
        Y = activation(Z)
        O = optimize_O(Y, T_train, Q, nodes, max_k, alpha, myu)
        S = np.dot(O, Y) 

        # for test data
        tmp_Z_test = np.dot(W, prev_Y_test)
        Z_test = normalize_Z(tmp_Z_test, Q)
        Y_test = activation(Z_test)
        S_test = np.dot(O, Y_test)
        cNME = compute_nme(S, T_train, m=X_train.shape[1])
        
        # break condition 
        if iter_j != 0 and not is_higer_threshold(cNME, oNME, eta_n):
            _logger.info("Break l'th layer")
            Y = cache_n["Y"]
            Y_test = cache_n["Y_test"]
            W = cache_n["W"]
            nodes = cache_n["nodes"]
            R = cache_n["R"]
            O = cache_n["O"]
            break
        
        # update previous result
        cache_n["W"]  = W
        cache_n["R"]  = R
        cache_n["Y"] = Y
        cache_n["Y_test"] = Y_test
        cache_n["nodes"] = nodes
        cache_n["O"] = O
        cache_n["NME"] = cNME
        oNME = cNME
        
        # Accuracy train
        train_accuracy = calculate_accuracy(S, T_train)
        train_NME = cNME
        #_logger.info("Train accuracy: {}".format(train_accuracy))
        #_logger.info("Train NME: {}".format(train_NME))

        #  Accuracy test
        test_NME = compute_nme(S_test, T_test, X_test.shape[1])
        test_accuracy = calculate_accuracy(S_test, T_test)
        _logger.info("Test accuracy: {}".format(test_accuracy))
        #_logger.info("Test NME: {}".format(test_NME))

        _logger.info("Add another node")

        # preserve performance
        el_train_NME_lists.append(train_NME)
        el_train_accuracy_lists.append(train_accuracy)
        el_test_NME_lists.append(test_NME)
        el_test_accuracy_lists.append(test_accuracy)
        
    else:
        _logger.info("Achieve the maximum of nodes")
    
    _logger.info("This layer has {} random nodes".format(nodes - 2 * Q))

    return W, Y, Y_test, nodes, O, R, el_train_accuracy_lists, el_train_NME_lists, el_test_accuracy_lists, el_test_NME_lists

def optimize_O(Y, T, Q, n, max_k, alpha, myu):
    """Optimize O by ADMM method"""
    epsilon = alpha * math.sqrt(2 * Q)
    mat_Q, Delta = np.zeros((Q, n)), np.zeros((Q, n))
    I = np.eye(n)
    for _ in range(max_k):
        O = np.dot(np.dot(T, Y.T) + 1 / myu * (mat_Q + Delta), np.linalg.inv(np.dot(Y, Y.T) + 1 / myu * I))
        mat_Q = project_function(O, Delta, epsilon)
        Delta = Delta + mat_Q - O
        
    return O

def project_function(O, Delta, epsilon):
    """Projection for ADMM"""
    x = O - Delta
    frobenius_norm = math.sqrt(np.sum(x**2))
    if frobenius_norm > epsilon:
        value = x * (epsilon/frobenius_norm)
    else:
        value = x
    
    return value
