# 단위행렬
# 대각행렬 중에 주대각선의 모든 요소가 1인 행렬
import numpy as np

m = np.eye(3, 3)
print(m)

m = np.eye(3, 3, k=1)
print(m)

m = np.eye(3, 3, k=-2)
print(m)
