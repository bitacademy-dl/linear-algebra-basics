# 행렬의 산술 연산: 곱셈
import numpy as np

m1 = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

m2 = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

m3 = m1 * m2
print(m3)

m4 = np.multiply(m1, m2)
print(m4)
