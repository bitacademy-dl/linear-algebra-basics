# 전치행렬(transpose matrix)

import numpy as np

m1 = np.arange(15).reshape(3, 5)
print(m1, m1.shape)

m2 = m1.T
print(m2, m2.shape)

m3 = np.transpose(m1)
print(m3, m3.shape)

m4 = np.swapaxes(m1, 0, 1)
print(m3, m3.shape)


