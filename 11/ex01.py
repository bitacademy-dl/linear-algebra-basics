# sigmoid graph
import numpy as np
from matplotlib import pyplot as plt


def f(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-10, 10, 0.1)
y = f(x)

fig, splt = plt.subplots()
splt.plot(x, y, 'k-')

plt.show()

