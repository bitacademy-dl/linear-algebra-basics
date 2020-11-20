# 다중선형회귀(수치미분)
import os
import sys
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import gradient_descent
except ImportError:
    print('Library Module Can Not Found')
import numpy as np


def mean_squares_error(x, data_training):
    data_x0, data_x1, data_y = data_training

    s = 0
    for i in range(len(data_x0)):
        data_y_hat = x[0] * data_x0[i] + x[1] * data_x1[i] + x[2]
        s += ((data_y_hat - data_y[i]) ** 2)
    e = s / len(data_x0)

    return e


# data
times = [2, 4, 6, 8]
ptimes = [0, 4, 2, 3]
scores = [81, 93, 91, 97]


x = np.array([0, 0, 0])
for i in range(5000):
    h = 1.0e-4
    h1 = mean_squares_error(np.array([x[0]+h, x[1], x[2]]), (times, ptimes, scores))
    h2 = mean_squares_error(np.array([x[0]-h, x[1], x[2]]), (times, ptimes, scores))
    dx0 = (h1-h2) / (2*h)

    h1 = mean_squares_error(np.array([x[0], x[1]+h, x[2]]), (times, ptimes, scores))
    h2 = mean_squares_error(np.array([x[0], x[1]-h, x[2]]), (times, ptimes, scores))
    dx1 = (h1-h2)/ (2 * h)

    h1 = mean_squares_error(np.array([x[0], x[1], x[2] + h]), (times, ptimes, scores))
    h2 = mean_squares_error(np.array([x[0], x[1], x[2] - h]), (times, ptimes, scores))
    dx2 = (h1-h2) / (2 * h)

    gradient = np.array([dx0, dx1, dx2])
    print(f'epoch={i + 1}, gradient={gradient}, x={x}')

    x = x - 0.01 * gradient

result = x


# 경사하강법
# result = gradient_descent(mean_squares_error, np.array([0., 0., 0.]), epoch=5000, data_training=(times, ptimes, scores))
# print(result)

# ==============================================================================================


# predict(inference)
x1_p = 2
x2_p = 2
y_p = result[0] * x1_p + result[1] * x2_p + result[2]
print(y_p)
