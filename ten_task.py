import numpy as np
import matplotlib.pyplot as plt

L = 1.0
T = 0.1
alpha = 1.0

Nx = 50
dx = L / Nx
x = np.linspace(0, L, Nx + 1)

dt_values = [0.1, 0.05, 0.025, 0.0125, 0.01, 0.005]
errors_dt = []

for dt in dt_values:
    Nt = int(T / dt)
    t = np.linspace(0, T, Nt + 1)

    sigma = alpha * dt / dx ** 2

    u = np.zeros((Nt + 1, Nx + 1))
    u[0, :] = np.sin(np.pi * x)

    u[:, 0] = 0
    u[:, -1] = 0

    A = np.zeros((Nx - 1, Nx - 1))
    B = np.zeros((Nx - 1, Nx - 1))

    for i in range(Nx - 1):
        if i > 0:
            A[i, i - 1] = -sigma / 2
            B[i, i - 1] = sigma / 2
        A[i, i] = 1 + sigma
        B[i, i] = 1 - sigma
        if i < Nx - 2:
            A[i, i + 1] = -sigma / 2
            B[i, i + 1] = sigma / 2

    for n in range(0, Nt):
        b = B @ u[n, 1:-1]
        u[n + 1, 1:-1] = np.linalg.solve(A, b)

    u_exact = np.zeros((Nt + 1, Nx + 1))
    for n in range(Nt + 1):
        u_exact[n, :] = np.sin(np.pi * x) * np.exp(-np.pi ** 2 * t[n])

    errors_dt.append(np.max(np.abs(u[-1, :] - u_exact[-1, :])))

plt.figure(figsize=(10, 6))
plt.loglog(dt_values, errors_dt, 'o-', label='Ошибка по dt')
plt.xlabel('Шаг по времени dt')
plt.ylabel('Максимальная ошибка')
plt.title('Сходимость по времени (dt)')
plt.grid(True)
plt.legend()
plt.show()

dt = 0.001
Nx_values = [20, 40, 80, 160]
errors_dx = []

for Nx in Nx_values:
    dx = L / Nx
    x = np.linspace(0, L, Nx + 1)

    sigma = alpha * dt / dx ** 2
    Nt = int(T / dt)
    t = np.linspace(0, T, Nt + 1)

    u = np.zeros((Nt + 1, Nx + 1))
    u[0, :] = np.sin(np.pi * x)

    u[:, 0] = 0
    u[:, -1] = 0

    A = np.zeros((Nx - 1, Nx - 1))
    B = np.zeros((Nx - 1, Nx - 1))

    for i in range(Nx - 1):
        if i > 0:
            A[i, i - 1] = -sigma / 2
            B[i, i - 1] = sigma / 2
        A[i, i] = 1 + sigma
        B[i, i] = 1 - sigma
        if i < Nx - 2:
            A[i, i + 1] = -sigma / 2
            B[i, i + 1] = sigma / 2

    for n in range(0, Nt):
        b = B @ u[n, 1:-1]
        u[n + 1, 1:-1] = np.linalg.solve(A, b)

    u_exact = np.zeros((Nt + 1, Nx + 1))
    for n in range(Nt + 1):
        u_exact[n, :] = np.sin(np.pi * x) * np.exp(-np.pi ** 2 * t[n])

    errors_dx.append(np.max(np.abs(u[-1, :] - u_exact[-1, :])))

h_values = [L / Nx for Nx in Nx_values]
plt.figure(figsize=(10, 6))
plt.loglog(h_values, errors_dx, 'o-', label='Ошибка по dx')
plt.xlabel('Шаг по пространству dx')
plt.ylabel('Максимальная ошибка')
plt.title('Сходимость по пространству (dx)')
plt.grid(True)
plt.legend()
plt.show()
