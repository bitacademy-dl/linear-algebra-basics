# 경사하강법(수치미분)
import os
import sys
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import method_least_squares, mean_squares_error, gradient_descent_linear_regression
except ImportError:
    print('Library Module Can Not Found')
import numpy as np


# data
times = [2, 4, 6, 8]
scores = [81, 93, 91, 97]

# 경사하강법
result = gradient_descent_linear_regression(mean_squares_error, np.array([0., 0.]), epoch=5000, data_training=(times, scores))
print(result)