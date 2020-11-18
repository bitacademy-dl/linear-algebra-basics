# 최소제곱법(Method of Least Squares)
import os
import sys
from pathlib import Path
try:
    sys.path.append(os.path.join(Path(os.getcwd()).parent, 'lib'))
    from common import method_least_squares
except ImportError:
    print('Library Module Can Not Found')
from matplotlib import pyplot as plt

# data
times = [2, 4, 6, 8]
scores = [81, 93, 91, 97]

a, b = method_least_squares(times, scores)
print(f'직선 y = {a}x + {b}')

# predict
scores_predict = [(a * t) + b for t in times]
print(scores_predict)

# 그래프
fig, splt = plt.subplots()
splt.scatter(times, scores)
splt.plot(times, scores_predict)
plt.show()
