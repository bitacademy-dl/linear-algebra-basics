# -logx & -logx(1-x)

from matplotlib import pyplot as plt
import numpy as np


def f1(x):
    return -np.log(x)


def f2(x):
    return -np.log(1-x)


data_x1 = np.arange(0.01, 1.5, 0.01)
data_y1 = f1(data_x1)

data_x2 = np.arange(-0.1, 1., 0.01)
data_y2 = f2(data_x2)

fig, splt = plt.subplots()
plt.axvline(x=0, color='r')
plt.axhline(y=0, color='r')

splt.plot(data_x1, data_y1)
splt.plot(data_x2, data_y2)
splt.set_xticks([])
splt.set_yticks([])

plt.show()
