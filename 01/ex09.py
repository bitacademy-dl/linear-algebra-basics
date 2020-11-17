# 단위행렬
import numpy as np

val = 4
v = np.array([val, val, val])

# 대각요소의 역수를 곱해서 단위행렬을 생성할 수 있다.
m = np.diag(v)
m = m * (1 / val)
print(m)
