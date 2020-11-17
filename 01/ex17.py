# 행렬과 스칼라 사칙연산

import numpy as np
m1 = np.array([
    [10, 20, 30],
    [40, 50, 90]
])

# 스칼라는 2차원 행렬로 변환(브로드캐스트)된 후, 요소별 사칙연산을 수행한다.
m2 = m1 + 10
print(m2)

m3 = m1 - 10
print(m3)

m4 = m1 * 10
print(m4)

m5 = m1 / 10
print(m5)

# add(), subtract(), mutiply(), divide() 함수도 마찬가지이다.
m6 = np.add(m1, 10)
print(m6)