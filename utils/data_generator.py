#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_generator.py
# @Author: Yin-tao Xu
# @Date  : 18-9-11
# @Desc  : a tiny tool to generate 2-d data pts. as metioned in
# LFD (Learning from data) 8-11 Example 8.4

import numpy as np

def generate_training_samples(n):
    """
    generate 2-d data pts, of which
    x_1 \in [0, 1], x_2 \in [-1, 1], y=f(x)=sign(x_2)
    :param n: the required quantity of the trainning sample
    :return: X, y
        X: a 2-d numpy array. X.shape = (n, 2,)
        y: a 1-d numpy array. y.shape = (n,)
    """
    X = np.zeros((n, 2), dtype=np.float)
    y = None

    # x_1 \in [0, 1], (for numerical issue, the pratical
    # range is [0, 1) )
    X[:, 0] = np.random.rand(n)
    # x_2 \in [-1, 1], (for numerical issue, the pratical
    # range is [-1, 1) )
    X[:, 1] = np.random.rand(n) * 2 - 1

    y = np.sign(X[:, 1])

    return X, y

if __name__ == "__main__":
    X, y = generate_training_samples(3)
    print(X, y)


