#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : SVM.py
# @Author: Yin-tao Xu
# @Date  : 18-9-11
# @Desc  : sealing the sklearn class into a function

# from sklearn.svm import LinearSVC
import numpy as np
from cvxopt import matrix, solvers

solvers.options["show_progress"] = False

class SVM():

    def __init__(self):
        self.has_fitted = False
        # threshold for identifying that this vector is support vector
        self.sp_v_threshold = 1e-8
        # list for supoort vector indexes
        self.sp_v_i = []

    def fit(self, X, y):
        assert len(X) > 0, "Please input at least one sample"

        # specify the shape
        n = len(X)
        dim = len(X[0])

        # initialize variables
        self.P = matrix(np.identity(dim + 1, dtype=np.float))
        self.q = matrix(np.zeros((dim + 1,), dtype=np.float))
        self.G = matrix(np.zeros((n, dim + 1), dtype=np.float))
        self.h = -matrix(np.ones((n,), dtype=np.float))

        # dump the data into variables
        # P
        self.P[0, 0] = 0
        # G = [y_1, --y_1 x_1^T--; y_2, --y_2 x_2^T--, ..., y_n, --y_n x_n^T]
        for i in range(n):
            self.G[i, 0] = -y[i]
            self.G[i, 1:] = -X[i, :] * y[i]

        sol = solvers.qp(self.P,
                         self.q,
                         self.G,
                         self.h
                         )

        # find the solution
        self.w = np.zeros(dim,) # weight
        self.b = sol["x"][0]
        n_s = 0.
        for i in range(1, dim + 1):
            self.w[i - 1] = sol["x"][i]
        for i in range(n):
            v = y[i] * (np.dot(self.w, X[i]) + self.b)
            if v < (1 + self.sp_v_threshold):
                self.sp_v_i.append(i)
        self.has_fitted = True
        self.intercept_ = np.array([self.b])
        self.coef_ = self.w.reshape(-1, 1)

    def predict(self, X):
        """
        predict the results due to the result of trainning
        :param X:
        :return: y_hat: numpy.array[n, ], with +1 positve sample while -1 negative sample
        """
        assert self.has_fitted, "Model has not been trained"
        return np.sign(np.dot(self.w, X.T) + self.b)

def linear_svm(X, y):
    # set the penalty term large enough to
    # achieve similar linear performance
    C = 1e8
    clf = SVM()
    clf.fit(X, y)
    return np.concatenate(
        (clf.intercept_.reshape(1,1), clf.coef_.reshape(-1, 1)),
        axis=0
    )

if __name__ == "__main__":
    from utils.data_generator import generate_training_samples
    from utils.eval_Eout import eval_Eout
    X, y = generate_training_samples(10)
    w_hat = linear_svm(X, y)
    print(eval_Eout(w_hat))
