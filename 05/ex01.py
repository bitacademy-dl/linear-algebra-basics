# 지수함수와 로그함수

from matplotlib import pyplot as plt
import numpy as np


def f1(x):
    return 2**x


def f2(x):
    return np.log2(x)


data_x1 = np.arange(-2, 5, 0.1)
data_y1 = f1(data_x1)
data_x2 = data_y1         # 역함수 관계
data_y2 = f2(data_x2)     # data_y2 = data_x1

fig, splt = plt.subplots(2, 1)
splt[0].plot(data_x1, data_y1)
splt[0].axvline(x=0, color='r')
splt[0].axhline(y=0, color='r')
splt[0].set_xticks([])
splt[0].set_yticks([])

splt[1].plot(data_x2, data_y2)
splt[1].axvline(x=0, color='r')
splt[1].axhline(y=0, color='r')
splt[1].set_xticks([])
splt[1].set_yticks([])

plt.show()
