import numpy as np

def bessel_function(m, x, N=1000, method='simpson'):
    t = np.linspace(0, np.pi, N + 1)
    f = np.cos(m * t - x * np.sin(t))

    if method == 'trapezoidal':
        h = np.pi / N
        return (h / 2) * (f[0] + 2 * np.sum(f[1:-1]) + f[-1]) / np.pi
    elif method == 'simpson':
        if N % 2 != 0:
            N += 1
        h = np.pi / N
        return (h / 3) * (f[0] + 4 * np.sum(f[1:-1:2]) + 2 * np.sum(f[2:-2:2]) + f[-1]) / np.pi

def bessel_derivative(m, x, N=1e7, method='simpson'):
    h = 1 / N
    return (bessel_function(m, x + h, N, method) - bessel_function(m, x - h, N, method)) / (2 * h)

def check_bessel_identity(x_values, N=1000, method='simpson'):
    for x in x_values:
        J0_prime = bessel_derivative(0, x, N=N, method=method)
        J1 = bessel_function(1, x, N=N, method=method)
        print(f"x = {x:.6f}: J'_0(x) + J_1(x) = {J0_prime + J1:.10e}")


x_values = np.linspace(0, 2 * np.pi, 10)
check_bessel_identity(x_values, N=1000000, method='simpson')
