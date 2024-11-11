import numpy as np
import matplotlib.pyplot as plt


a, b = -np.pi / 2, np.pi / 2
N = 100
h = (b - a) / (N - 1)
x = np.linspace(a, b, N)

f = np.cos(x)

A = np.zeros((N, N))
B = np.zeros(N)

for i in range(1, N - 1):
    A[i, i - 1] = 1 / h ** 2
    A[i, i] = -2 / h ** 2
    A[i, i + 1] = 1 / h ** 2
    B[i] = f[i]

A[0, 0] = 1
A[-1, -1] = 1
B[0] = 0
B[-1] = 0


def progonka(A, B):
    N = len(B)
    alpha = np.zeros(N)
    beta = np.zeros(N)

    alpha[1] = -A[0, 1] / A[0, 0]
    beta[1] = B[0] / A[0, 0]

    for i in range(1, N - 1):
        denominator = A[i, i] + A[i, i - 1] * alpha[i]
        alpha[i + 1] = -A[i, i + 1] / denominator
        beta[i + 1] = (B[i] - A[i, i - 1] * beta[i]) / denominator

    y = np.zeros(N)
    y[-1] = (B[-1] - A[-1, -2] * beta[-2]) / (A[-1, -1] + A[-1, -2] * alpha[-2])

    for i in range(N - 2, -1, -1):
        y[i] = alpha[i + 1] * y[i + 1] + beta[i + 1]

    return y


y = progonka(A, B)
#порядок сходимости
plt.plot(x, y, label='Численное решение')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.title('Решение разностного уравнения методом прогонки')
plt.show()
