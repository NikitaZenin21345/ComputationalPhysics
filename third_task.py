import numpy as np

def f(x):
    return 1 / (1 + x**2)

def left_riemann(f, a, b, N):
    h = (b - a) / N
    x = np.linspace(a, b - h, N)
    return np.sum(f(x)) * h

def right_riemann(f, a, b, N):
    h = (b - a) / N
    x = np.linspace(a + h, b, N)
    return np.sum(f(x)) * h

def trapezoidal(f, a, b, N):
    h = (b - a) / N
    x = np.linspace(a, b, N+1)
    return (h/2) * (f(a) + 2*np.sum(f(x[1:-1])) + f(b))

def simpsons(f, a, b, N):
    if N % 2 != 0:
        N += 1
    h = (b - a) / N
    x = np.linspace(a, b, N+1)
    return (h/3) * (f(a) + 4*np.sum(f(x[1:-1:2])) + 2*np.sum(f(x[2:-2:2])) + f(b))

a, b = -1, 1
N = 1000

left_sum = left_riemann(f, a, b, N)
right_sum = right_riemann(f, a, b, N)
trap_sum = trapezoidal(f, a, b, N)
simpson_sum = simpsons(f, a, b, N)
#analytic solution pi / 2
analytic = np.pi / 2
print(f"Analytic solution: {analytic}")
print(f"Left Riemann Sum: {left_sum - analytic}")
print(f"Right Riemann Sum: {right_sum - analytic}")
print(f"Trapezoidal Rule: {trap_sum - analytic}")
print(f"Simpson's Rule: {simpson_sum - analytic}")
print(f"-----------------------------------------------")
count = 4

while count > 0:
    N *= 2
    left_sum = left_riemann(f, a, b, N)
    right_sum = right_riemann(f, a, b, N)
    trap_sum = trapezoidal(f, a, b, N)
    simpson_sum = simpsons(f, a, b, N)
    print(f"Left Riemann Sum: {left_sum - analytic}")
    print(f"Right Riemann Sum: {right_sum - analytic}")
    print(f"Trapezoidal Rule: {trap_sum - analytic}")
    print(f"Simpson's Rule: {simpson_sum - analytic}")
    print(f"-----------------------------------------------")
    count -= 1

def gauss(t):
    return np.exp(-t**2)

def erf(x, N=1000):
    constant = 2 / np.sqrt(np.pi)
    integral_value = simpsons(gauss, 0, x, N)
    return constant * integral_value

x_value = 1
erf_value = erf(x_value)

print(f"erf({x_value}) = {erf_value}")