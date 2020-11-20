import numpy as np


def mean_squares_error(x, data_in, data_out):
    # if data_in.ndim == 1:
    #      data_in = data_in[np.newaxis, :]
    e = np.mean(((x[:-1] @ (data_in[np.newaxis, :] if data_in.ndim == 1 else data_in) + x[-1:]) - data_out)**2)
    return e


# x가 2개 이상인 경우
# y = a0x0 + a1x1 + b
times = np.array([2, 4, 6, 8])
ptimes = np.array([0, 4, 2, 3])
scores = np.array([81, 93, 91, 97])
x = np.array([2., 3., 4.])

r = mean_squares_error(x, np.array([times, ptimes]), scores)
print(r)


# x가 1개 이상인 경우
# y = ax + b
times = np.array([2, 4, 6, 8])
scores = np.array([81, 93, 91, 97])
x = np.array([2., 4.])

r = mean_squares_error(x, times, scores)
print(r)


def f(x):
    return np.sum(x**2)

