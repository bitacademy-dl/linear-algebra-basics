#영행렬
import numpy as np

# 직접 형상을 전달
m = np.zeros((3, 3))
print(m)

m1 = np.arange(1, 10).reshape(3, 3)
m2 = np.zeros_like(m1)
print(m2)


