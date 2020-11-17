# 행렬의 산술 연산: 나눗셈
import numpy as np

m1 = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

m2 = np.array([
    [10, 20, 30],
    [40, 50, 60]
])

m3 = m2 / m1
print(m3)

m4 = np.divide(m2, m1)
print(m4)
