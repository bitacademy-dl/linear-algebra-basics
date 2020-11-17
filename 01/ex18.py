# dot(@)연산
import numpy as np

m1 = np.array([
    [1, 2],
    [3, 4],
    [5, 6]
])
print(m1.shape)

m2 = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8]
])
print(m2.shape)

# (3 x 2) @ (2 x 4)   ->  (3 x 4)
m3 = np.dot(m1, m2)
print(m3)
print(m3.shape)

# 연산자 @ 사용
m4 = m1 @ m2
print(m4)
print(m4.shape)

# (2 x 4) @ (3 x 2) -> 오류!
# m5 = np.dot(m2, m1)



