# sigmoid 함수의 특징 II
import numpy as np
from matplotlib import pyplot as plt


def f(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-10, 10, 0.1)

fig, splt = plt.subplots()
for b in np.arange(-5, 5):
    y = f(x + b)
    splt.plot(x, y)

plt.show()

