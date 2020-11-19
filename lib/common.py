from inspect import signature

import numpy as np


# 수치미분(x변수가 하나일 때....)
def numerical_diff(f, x):
    h = 1e-4
    dx = (f(x+h) - f(x-h)) / (2 * h)

    return dx


# 수치편미분(x변수가 여러 개 일때..)
def numerical_partial_diff(f, x, data_training=None):
    """
    return 변수 x(벡터,1차원 numpy array)에 대한 편미분 결과(벡터, 1차원 numpy array) 반환
    : param f: 손실함수
    : pram x : 변수(벡터, 1차원 numpy array)
    """
    h = 1e-4
    dx = np.zeros_like(x)

    for i in range(x.size):
        tmp = x[i]

        x[i] = tmp + h
        h1 = f(x) if len(signature(f).parameters) == 1 else f(x, data_training)

        x[i] = tmp - h
        h2 = f(x) if len(signature(f).parameters) == 1 else f(x, data_training)

        dx[i] = (h1 - h2) / (2 * h)
        x[i] = tmp

    return dx


# 기울기 = 수치편미분
numerical_gradient = numerical_partial_diff





# 경사하강법: no training data
def gradient_descent(f, x, lr=0.01, epoch=100):
    for i in range(epoch):
        gradient = numerical_gradient(f, x)
        # 출력
        print(f'epoch={i+1}, gradient={gradient}, x={x}')
        x -= lr * gradient

    return x





# 최소제곱법
# 여러 점에서 직선의 기울기 구하기
def method_least_squares(x, y):
    mx = sum(x)/len(x)
    my = sum(y)/len(y)

    s1 = 0
    for i in range(len(x)):
        s1 += (x[i] - mx) * (y[i] - my)

    s2 = 0
    for i in range(len(x)):
        s2 += (x[i] - mx)**2

    mls_a = s1 / s2
    mls_b = my - (mx * mls_a)

    return mls_a, mls_b

























# 경사하강법 구현2 - 선형회귀
def gradient_descent_linear_regression(f, x, lr=0.01, epoch=100, data_training=None):
    for i in range(epoch):
        gradient = numerical_gradient_training(f, x, data_training)
        # 출력
        print(f'epoch={i+1}, gradient={gradient}, x={x}')
        x -= lr * gradient

    return x





