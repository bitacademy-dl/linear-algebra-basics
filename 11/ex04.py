#Logistic Regression(로지스틱 회귀, 수치미분)
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loss(x, data_in, data_out):
    e = data_out * np.log(sigmoid(x[0] * data_in + x[1])) + (1-data_out) * np.log(1-sigmoid(x[0] * data_in + x[1]))
    return -1 * np.mean(e)


# data
times = [2, 4, 6, 8, 10, 12, 14]
passed = [0, 0, 0, 1, 1, 1, 1]

