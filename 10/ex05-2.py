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

# 경사하강법
x = np.array([0, 0, 0])
data_training = (times, ptimes, scores)

for i in range(5000):
    h = 1.0e-4
    h1 = mean_squares_error(np.array([x[0]+h, x[1], x[2]]), data_training)
    h2 = mean_squares_error(np.array([x[0]-h, x[1], x[2]]), data_training)
    dx0 = (h1-h2) / (2*h)

    h1 = mean_squares_error(np.array([x[0], x[1]+h, x[2]]), data_training)
    h2 = mean_squares_error(np.array([x[0], x[1]-h, x[2]]), data_training)
    dx1 = (h1-h2) / (2 * h)

    h1 = mean_squares_error(np.array([x[0], x[1], x[2] + h]), data_training)
    h2 = mean_squares_error(np.array([x[0], x[1], x[2] - h]), data_training)
    dx2 = (h1-h2) / (2 * h)

    gradient = np.array([dx0, dx1, dx2])
    print(f'epoch={i + 1}, gradient={gradient}, x={x}')

    x = x - 0.01 * gradient

# predict(inference)
x1_p = 4
x2_p = 0
y_p = x[0] * x1_p + x[1] * x2_p + x[2]
print(f'공부를 {x1_p}시간 하고 과외를 {x2_p}시간 받았을  때, 받을 수 있는 점수는 {y_p}점')
