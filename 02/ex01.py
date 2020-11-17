# y = ax + b 일차 방정식 결정하기
from matplotlib import pyplot as plt

data_x = [2, 6]
data_y = [81, 91]

# 기울기
a = (data_y[1] - data_y[0]) / (data_x[1] - data_x[0])
b = data_y[1] - a * data_x[1]

# 결과
print(f'직선 y = {a}x + {b}')

# 그래프
fig, splt = plt.subplots()
x = [2, 4, 6, 8]
y1 = [81, 93, 91, 97]
y2 = [a * i + b for i in x]
splt.scatter(x, y1)
splt.plot(x, y2, 'r-')

plt.show()


