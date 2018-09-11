#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : comparison.py
# @Author: Yin-tao Xu
# @Date  : 18-9-11
# @Desc  : Compare the result of PLA and SVM
from utils.data_generator import generate_training_samples
from utils.eval_Eout import eval_Eout
from SVM import linear_svm
from PLA import PLA
import matplotlib.pyplot as plt
from multiprocessing import Process
import seaborn as sns
import numpy as np

def compare():
    # quantity of the sample
    N = 20
    # count of the trials
    count_of_trails = 4000

    X, y = generate_training_samples(N)
    # out-of-sample Error for linear hard-margin SVM
    E_out_SVM = eval_Eout(linear_svm(X, y))
    # out-of-sample Error for pocket PLA(perceptron learning algorithm)
    E_outs_PLA = []
    for _ in range(count_of_trails):
        random_i = np.random.permutation(X.shape[0])
        X = X[random_i]
        y = y[random_i]
        E_outs_PLA.append(eval_Eout(PLA(X, y)))
    return E_out_SVM, np.array(E_outs_PLA)

if __name__ == "__main__":
    E_out_SVM, E_outs_PLA = compare()
    ax = sns.distplot(E_outs_PLA, kde=False)
    plt.axvline(E_out_SVM, color="red", lw=1.2)
    plt.text(E_out_SVM + (ax.axis()[1] - ax.axis()[0]) / 10,
             0.8 * ax.axis()[3],
             r'$E_{out}(SVM)$',
             fontsize=20
             )
    plt.show()
