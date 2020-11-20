# sigmoid 함수의 특징 I
import numpy as np
from matplotlib import pyplot as plt


def f(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-10, 10, 0.1)

fig, splt = plt.subplots()
for a in np.arange(10):
    y = f(a * x)
    splt.plot(x, y)

plt.show()

