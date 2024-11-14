import numpy as np
import matplotlib.pyplot as plt

L = 1.0
T = 0.1
nx = 50
nt = 1000
dx = L / (nx - 1)
dt = T / nt
alpha = dt / dx**2

if alpha > 0.5:
    print("решение может быть неустойчивым")

x = np.linspace(0, L, nx)
t = np.linspace(0, T, nt)

u = np.sin(np.pi * x / L)
u_analytical = np.zeros_like(u)

A = np.zeros((nx-2, nx-2))
B = np.zeros((nx-2, nx-2))

for i in range(nx-2):
    for j in range(nx-2):
        if i == j:
            A[i,j] = 1 + alpha
            B[i,j] = 1 - alpha
        elif abs(i - j) == 1:
            A[i,j] = -alpha / 2
            B[i,j] = alpha / 2

u_num = u.copy()

for n in range(nt):
    b = B.dot(u_num[1:-1])
    u_new = np.linalg.solve(A, b)
    u_num[1:-1] = u_new

u_analytical = np.sin(np.pi * x / L) * np.exp(- (np.pi / L)**2 * T)

plt.figure(figsize=(8,6))
plt.plot(x, u_num, label='Численное решение (Кранк-Николсон)')
plt.plot(x, u_analytical, '--', label='Аналитическое решение')
plt.xlabel('x')
plt.ylabel('u(x, T)')
plt.legend()
plt.title('Сравнение')
plt.grid(True)
plt.show()

errors = []
dx_values = []
nx_values = [10, 20, 40, 80, 160]
for nx in nx_values:
    dx = L / (nx - 1)
    dt = alpha * dx**2
    nt = int(T / dt)
    x = np.linspace(0, L, nx)
    u = np.sin(np.pi * x / L)
    u_num = u.copy()

    A = np.zeros((nx-2, nx-2))
    B = np.zeros((nx-2, nx-2))
    for i in range(nx-2):
        for j in range(nx-2):
            if i == j:
                A[i,j] = 1 + alpha
                B[i,j] = 1 - alpha
            elif abs(i - j) == 1:
                A[i,j] = -alpha / 2
                B[i,j] = alpha / 2

    for n in range(nt):
        b = B.dot(u_num[1:-1])
        u_new = np.linalg.solve(A, b)
        u_num[1:-1] = u_new

    u_analytical = np.sin(np.pi * x / L) * np.exp(- (np.pi / L)**2 * T)
    error = np.max(np.abs(u_num - u_analytical))
    errors.append(error)
    dx_values.append(dx)

plt.figure(figsize=(8,6))
plt.loglog(dx_values, errors, '-o')
plt.xlabel('dx')
plt.ylabel('Максимальная ошибка')
plt.title('Сходимость схемы')
plt.grid(True)
plt.show()
