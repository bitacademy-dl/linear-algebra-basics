# 일차방정식 그래프
from matplotlib import pyplot as plt
import numpy as np


# a = 6, b = 20
def f(x):
    return 6 * x + 20


data_x = np.arange(-10, 10, 0.1)
data_y = f(data_x)

fig, splt = plt.subplots()
plt.axvline(x=0, color='r')
plt.axhline(y=0, color='r')

splt.plot(data_x, data_y)
splt.set_xticks([-10, -5, 0, 5, 10])
splt.set_yticks([-100, -50, 0, 50, 100])

plt.show()




