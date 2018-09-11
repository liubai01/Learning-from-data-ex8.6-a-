#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : PLA.py
# @Author: Yin-tao Xu
# @Date  : 18-9-11
# @Desc  : Implementation of the PLA algorithm
import numpy as np

def PLA(X, y, max_update=100):
    n_train = X.shape[0]
    X_aug = np.concatenate((np.ones((n_train, 1)), X), axis=1)
    w = np.zeros((X_aug.shape[1], 1))
    max_acc = 0
    for _ in range(max_update):
        pred = np.matmul(X_aug, w)
        pred = np.sign(pred) - (pred == 0)
        pred = pred.reshape(-1)

        for i, j in enumerate(pred != y):
            if j == 1:
                break
            if i == X.shape[0] - 1:
                return w
        w += y[i] * X_aug[i, :].reshape(-1, 1)
    return w

if __name__ == "__main__":
    from utils.data_generator import generate_training_samples
    from utils.eval_Eout import eval_Eout
    X, y = generate_training_samples(10)
    w_hat = PLA(X, y)
    E_out = eval_Eout(w_hat)
    print(E_out)
