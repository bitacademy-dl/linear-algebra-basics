# 기울기(gradient)
import os
import sys
from pathlib import Path

import numpy as np
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import numerical_gradient
except ImportError:
    print('Library Module Can Not Found')


def f(x):
    return np.sum(x**2, axis=0)


# 함수 테스트
# print(f(np.array([3., 4.])))

gra1 = numerical_gradient(f, np.array([3., 4.]))
gra2 = numerical_gradient(f, np.array([-1., -1.5]))
gra3 = numerical_gradient(f, np.array([-0.25, -0.25]))

print(gra1, gra2, gra3)


