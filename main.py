# -*- coding: utf-8 -*-

import logging 
import argparse
from multi_layer_ssfn import ssfn
from backprop import back_propagation
from make_dataset_helper import *


def define_parser():
    parser = argparse.ArgumentParser(description="Run progressive learning")
    parser.add_argument("--data", default="mnist", help="Input dataset available as the paper shows")
    parser.add_argument("--lambda_ls", type=float, default=1, help="Reguralized parameters on the least-square problem")
    parser.add_argument("--myu", type=int, default=1, help="Parameter for ADMM")
    parser.add_argument("--max_k", type=int, default=100, help="Iteration number of ADMM")
    parser.add_argument("--alpha", type=int, default=2, help="Parameter for reguralization")
    parser.add_argument("--max_n", type=int, default=1000, help="Max number of random nodes on each layer")
    parser.add_argument("--eta_n", type=float, default=0.005, help="Threshold of nodes")
    parser.add_argument("--eta_l", type=float, default=0.1, help="Threshold of layers")
    parser.add_argument("--max_l", type=int, default=20, help="Max number of layers")
    parser.add_argument("--delta", type=int, default=50, help="Number of random nodes to add once")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for back propagation")
    parser.add_argument("--iteration_num", type=int, default=1000, help="Iteration number of back propagation") 
    args = parser.parse_args()
    return args

def define_logger():
    _logger = logging.getLogger(__name__)
    logging.basicConfig(
    level=logging.INFO, format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s")
    return _logger

def define_dataset(args):
    if args.data == "vowel":
        X_train,X_test, T_train,  T_test  = prepare_vowel()
    elif args.data == "yale":
        X_train,X_test, T_train,  T_test  = prepare_yale()
    elif args.data == "ar":
        X_train,X_test, T_train,  T_test  = prepare_ar()
    elif args.data == "scene":
        X_train,X_test, T_train,  T_test  = prepare_scene()
    elif args.data == "caltech":
        X_train,X_test, T_train,  T_test  = prepare_caltech()
    elif args.data == "letter":
        X_train,X_test, T_train,  T_test  = prepare_letter()
    elif args.data == "norb":
        X_train,X_test, T_train,  T_test  = prepare_norb()
    elif args.data == "shuttle":
        X_train,X_test, T_train,  T_test  = prepare_shuttle()
    elif args.data == "mnist":
        X_train,X_test, T_train,  T_test  = prepare_mnist()
    elif args.data == "cifar10":
        X_train,X_test, T_train,  T_test  = prepare_cifar10()
    elif args.data == "cifar100":
        X_train,X_test, T_train,  T_test  = prepare_cifar100()
    elif args.data == "satimage":
        X_train,X_test, T_train,  T_test  = prepare_satimage()
    return X_train, X_test, T_train, T_test

def set_hparameters(args):
    ssfn_hparameters = {"data": args.data, "lambda": args.lambda_ls, "myu": args.myu, \
            "max_k": args.max_k, "alpha": args.alpha, "max_n": args.max_n, \
            "eta_n": args.eta_n, "eta_l": args.eta_l, "max_l": args.max_l,\
            "delta": args.delta}
    bp_hparameters = {"data": args.data, "iteration_num": args.iteration_num, \
            "learning_rate": args.learning_rate}
    return ssfn_hparameters, bp_hparameters

def main():
    args = define_parser()
    _logger = define_logger()
    X_train, X_test, T_train, T_test = define_dataset(args)
    _logger.info("The dataset we use is {}".format(args.data))

    ssfn_hparameters, bp_hparameters = set_hparameters(args)
    
    _logger.info("Start construct SSFN")
    ssfn(_logger, X_train, X_test, T_train, T_test, ssfn_hparameters)
    
    _logger.info("Start Back Propagation")
    _logger.info("Learning rate={}".format(args.learning_rate))
    back_propagation(_logger, X_train, X_test, T_train, T_test, bp_hparameters)


if __name__ == '__main__':
    main()