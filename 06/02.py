# 수치미분(Numerical Diffirentiation) VS 해석미분(Analytic Diffirentiation)

def f(x):
    return 20*(x-2)**2+500


def analytic_diff(x):
    return 40 * x - 80


def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x)) / h


print(f'Numerical Diffirentiation Value:{numerical_diff(f, 5)}')
print(f'Analytic Diffirentiation Value:{analytic_diff(5)}')

