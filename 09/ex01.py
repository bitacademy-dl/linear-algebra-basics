import os
import sys
from pathlib import Path
import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import gradient_descent
except ImportError:
    print('Library Module Can Not Found')


def f(x):
    return np.sum(x**2, axis=0)


gradient_descent(f, np.array([-3., 4.]), lr=0.1)
# gradient_descent(f, np.array([-3., 4.]), lr=10)
# gradient_descent(f, np.array([-3., 4.]), lr=0.001)