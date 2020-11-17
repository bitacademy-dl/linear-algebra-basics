# Matrix(Tensor 2) 확인

import numpy as np

"""
1. 행렬은 두 개의 축을 가지며 2차원(Tensor 2) 이다.
2. 수평 방향을 행(row), 수직방향을 열(column) 이라 한다.
"""
m = np.array([[50, 10, 80],
              [55, 11, 88]])
print(m.ndim, m.shape)
