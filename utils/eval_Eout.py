#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : eval_Eout.py
# @Author: Yin-tao Xu
# @Date  : 18-9-11
# @Desc  : 
from .data_generator import generate_training_samples
import numpy as np

def eval_Eout(w_hat, n_samples=5000):
    """
    Estimate E_out by enough repeated time
    :param w_hat:
    :param n_samples:
    :return: float, out-of-sample error
    """
    X, y = generate_training_samples(n_samples)
    n_train = X.shape[0]
    X_aug = np.concatenate((np.ones((n_train, 1)), X), axis=1)
    pred = np.matmul(X_aug, w_hat).reshape(-1)
    pred = np.sign(pred) - (pred == 0)
    return 1 - np.count_nonzero(pred == y) / n_train
