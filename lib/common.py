import numpy as np


def numerical_gradient(f, x):
    h = 1e-4
    gradient = np.zeros_like(x)

    for i in range(x.size):
        tmp = x[i]

        x[i] = tmp + h
        h1 = f(x)
        x[i] = tmp - h
        h2 = f(x)
        gradient[i] = (h1 - h2) / (2 * h)

        x[i] = tmp

    return gradient
