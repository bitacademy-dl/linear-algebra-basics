# 평균제곱오차 테스트
import os
import sys
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import method_least_squares, mean_squares_error
except ImportError:
    print('Library Module Can Not Found')
import numpy as np

# data
times = [2, 4, 6, 8]
scores = [81, 93, 91, 97]

a, b = method_least_squares(times, scores)
print(f'직선 y = {a}x + {b}')

print(f'오차(평균제곱오차):{mean_squares_error(np.array([a, b]), (times, scores))}')

