# -*- coding: utf-8 -*-

import numpy as np
import logging
from function import *
from optimize_output import optimize_each_layer
from optimize_wl import optimize_wl
import json
import os


def save_parameters(outputs, parameters_path, data, n_lists):
    for key, value in outputs.items():
        np.save(parameters_path + data +'_'+key + ".npy", value)

    with open(parameters_path + data+"_"+'n_lists.json','w') as f: 
        json.dump(n_lists, f, ensure_ascii=False)

def ssfn(_logger, X_train, X_test, T_train, T_test, ssfn_hparameters):
    """Build self size-estimating Feed-forward network(SSFN)"""
    # define variables
    Q = T_train.shape[0]
    max_l = ssfn_hparameters["max_l"]
    eta_l = ssfn_hparameters["eta_l"]
    data = ssfn_hparameters["data"]
    
    # initialize collections
    n_lists = []
    outputs = {}
    train_accuracy_lists = []
    test_accuracy_lists = []
    train_NME_lists = []
    test_NME_lists = []
    
    # create a necessary directory
    parameters_path = "./parameters/"
    create_directory(parameters_path)

    _logger.info("Begin to optimize Wl")
    Wl, train_accuracy, test_accuracy, train_NME, test_NME  = \
        optimize_wl(X_train, X_test, T_train, T_test, ssfn_hparameters)
    #_logger.info("Train NME: {}".format(train_NME))
    #_logger.info("Test NME: {}".format(test_NME))
    #_logger.info("Train Accuracy: {}".format(train_accuracy))
    _logger.info("Test Accuracy: {}".format(test_accuracy))
    _logger.info('Finish optimizing Wl')

    # preserve the optimized cost and parameter
    outputs["Wl"] = Wl
    train_accuracy_lists.append(train_accuracy)
    test_accuracy_lists.append(test_accuracy)
    train_NME_lists.append(train_NME)
    test_NME_lists.append(test_NME)

    old_train_NME = train_NME # initialize old train NME
    for lth_layer in range(1, max_l + 1):
        _logger.info("Begin to optimize {}th layer".format(lth_layer))
        W, Y, Y_test, nodes, O, R, el_train_accuracy_lists, \
        el_train_NME_lists, el_test_accuracy_lists, el_test_NME_lists = \
                 optimize_each_layer(_logger, X_train, X_test, T_train, T_test, outputs, n_lists, lth_layer, ssfn_hparameters)

        _logger.info("Finish {}th layer".format(lth_layer))
        current_train_NME = el_train_NME_lists[-1]
        if not is_higer_threshold(current_train_NME, old_train_NME, eta_l):
            # break condition
            break
        
        # preserve optimized parameters
        n_lists.append(nodes)
        outputs["W" +str(lth_layer)] = W
        outputs["R" + str(lth_layer)] = R
        outputs["Y" +str(lth_layer)] = Y
        outputs["O"+str(lth_layer)] = O
        outputs["Y_test"+str(lth_layer)] = Y_test

        # preserve the lth layers performance
        train_accuracy_lists.extend(el_train_accuracy_lists)
        test_accuracy_lists.extend(el_test_accuracy_lists)
        train_NME_lists.extend(el_train_NME_lists)
        test_NME_lists.extend(el_test_NME_lists)

        # update old train NME
        old_train_NME = current_train_NME
        _logger.info("Add another layer")
    
    # if proceeding without break this loop
    else:
        _logger.info("Achieve the maximum layer number")
        
    _logger.info("Finish constructing neural network")
    
    _logger.info("The final result")
    #_logger.info("Train accuracy: {}".format(train_accuracy_lists[-1]))
    _logger.info("Test accuracy: {}".format(test_accuracy_lists[-1]))
    #_logger.info("Train NME: {}".format(train_NME_lists[-1]))
    #_logger.info("Test NME: {}".format(test_NME_lists[-1]))
    _logger.info("Node Information: {}.".format(n_lists))
    
    #_logger.info("Plot network architecture and performance as SSN constructs")
    #plot_data(Q, n_lists, train_NME_lists, test_NME_lists, train_accuracy_lists, test_accuracy_lists, data, ssfn_hparameters)

    _logger.info("Preserve optimized parameters for back propagation")
    save_parameters(outputs, parameters_path, data, n_lists)