# -*- coding: utf-8 -*-

import numpy as np
import logging
from function import *


def optimize_wl(X_train, X_test, T_train, T_test, pln_hparameters):
    """Solve the optimization problem as regularized least-squares"""
    P = X_train.shape[0]
    m = X_train.shape[1]
    lambd_ls = pln_hparameters["lambda"]
    
    if P < m:
        W_ls = np.dot(np.dot(T_train,X_train.T), np.linalg.inv(np.dot(X_train,X_train.T) + lambd_ls * np.eye(P))).astype(np.float32)
    else:
        W_ls = np.dot(T_train,np.linalg.inv(np.dot(X_train.T, X_train) + lambd_ls * np.eye(m))).dot(X_train.T)

    T_hat_train = np.dot(W_ls, X_train)
    T_hat_test = np.dot(W_ls, X_test)
    train_accuracy = calculate_accuracy(T_hat_train, T_train)
    test_accuracy = calculate_accuracy(T_hat_test, T_test)
    train_NME = compute_nme(T_hat_train, T_train, T_train.shape[1])
    test_NME = compute_nme(T_hat_test, T_test, T_test.shape[1])
    
    return W_ls,train_accuracy, test_accuracy, train_NME, test_NME