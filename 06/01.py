# 수치미분 (Numerical Diffirentiation)
# 이차함수 y=20(x-2)^2 + 500

def f(x):
    return 20*(x-2)**2+500


def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x)) / h


print(f'Diffirentiation Value:{numerical_diff(f, 2.)}')
print(f'Diffirentiation Value:{numerical_diff(f, 1.9)}')
