# 다중선형회귀(해석미분)
import numpy as np


def analytic_gradient(x, data_training):
    dx = np.zeros_like(x)

    n = len(data_training[0])
    error = x[0] * data_training[0] + x[1] * data_training[1] + x[2] - data_training[2]

    dx[0] = (2/n) * np.sum(error * data_training[0])
    dx[1] = (2/n) * np.sum(error * data_training[1])
    dx[2] = (2/n) * np.sum(error)

    return dx


# data
times = [2, 4, 6, 8]
ptimes = [0, 4, 2, 3]
scores = [81, 93, 91, 97]

# 경사 하강법
x = np.array([0., 0., 0.])
for i in range(5000):
    gradient = analytic_gradient(x, data_training=(np.array(times), np.array(ptimes), np.array(scores)))
    # 출력
    print(f'epoch={i+1}, gradient={gradient}, x={x}')
    x -= 0.01 * gradient


# predict(inference)
x1_p = 4
x2_p = 0
y_p = x[0] * x1_p + x[1] * x2_p + x[2]
print(f'공부를 {x1_p}시간 하고 과외를 {x2_p}시간 받았을  때, 받을 수 있는 점수는 {y_p}점')