# 수치미분 (Numerical Diffirentiation)
# 이차함수 y=20(x-2)^2 + 500
import os
import sys
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import numerical_diff
except ImportError:
    print('Library Module Can Not Found')
import numpy as np


def f(x):
    return 20*(x-2)**2+500


print(f'Diffirentiation Value:{numerical_diff(f, 2.)}')
print(f'Diffirentiation Value:{numerical_diff(f, 1.9)}')
