# 편미분(Partial Diffirentiation)

def f(x0):
    return x0**2 + 4**2


def numerical_diff(f, x0):
    h = 1.0e-4
    return (f(x0 + h) - f(x0 - h)) / (2 * h)


def analytic_diff(x0):
    return 2 * x0


# (x0, x1) = (3, 4)
print(f'Numerical Partial Diffierentiation Value:{numerical_diff(f, 3)}')
print(f'Analytic Partial Diffierentiation Value:{analytic_diff(3)}')

# (x0, x1) = (1, 4)
print(f'Numerical Partial Diffierentiation Value:{numerical_diff(f, 1)}')
print(f'Analytic Partial Diffierentiation Value:{analytic_diff(1)}')

# (x0, x1) = (2, 4)
print(f'Numerical Partial Diffierentiation Value:{numerical_diff(f, 2)}')
print(f'Analytic Partial Diffierentiation Value:{analytic_diff(2)}')
