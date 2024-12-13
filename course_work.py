import numpy as np
import matplotlib.pyplot as plt

x_min, x_max = -5, 10
t_max = 1
x_points = 200
t_points = 500

dx = (x_max - x_min) / (x_points - 1)
dt = t_max / t_points

x = np.linspace(x_min, x_max, x_points)
t = np.linspace(0, t_max, t_points)

u_init = np.exp(-x**2 / 2)

def explicit_scheme(u, dx, dt):
    u_new = np.copy(u)
    for i in range(1, len(u) - 1):
        u_new[i] = u[i] - dt / dx * u[i] * (u[i] - u[i-1])
    return u_new

def lax_wendroff_scheme(u, dx, dt):
    u_new = np.copy(u)
    for i in range(1, len(u) - 1):
        u_new[i] = (
            u[i]
            - dt / (2 * dx) * u[i] * (u[i+1] - u[i-1])
            + dt**2 / (2 * dx**2) * u[i] * (u[i+1] - 2 * u[i] + u[i-1])
        )
    return u_new

u_explicit = np.copy(u_init)
u_explicit_results = np.zeros((t_points, x_points))
u_explicit_results[0, :] = u_explicit

for n in range(1, t_points):
    u_explicit = explicit_scheme(u_explicit, dx, dt)
    u_explicit_results[n, :] = u_explicit

u_lax_wendroff = np.copy(u_init)
u_lax_wendroff_results = np.zeros((t_points, x_points))
u_lax_wendroff_results[0, :] = u_lax_wendroff

for n in range(1, t_points):
    u_lax_wendroff = lax_wendroff_scheme(u_lax_wendroff, dx, dt)
    u_lax_wendroff_results[n, :] = u_lax_wendroff

def exact_solution(x, t):
    u_exact = np.zeros_like(x)
    for i, xi in enumerate(x):
        u_prev = np.exp(-xi**2 / 2)
        for _ in range(10):
            u_prev = np.exp(-((xi - u_prev * t)**2) / 2)
        u_exact[i] = u_prev
    return u_exact


u_exact = exact_solution(x, t_max)
error_explicit = np.abs(u_explicit_results[-1, :] - u_exact)
error_lax_wendroff = np.abs(u_lax_wendroff_results[-1, :] - u_exact)

plt.figure(figsize=(14, 12))

plt.subplot(3, 2, 1)
plt.plot(x, u_exact, 'k-', label='Точное решение', linewidth=2)
plt.plot(x, u_explicit_results[-1, :], 'r--', label='Явная схема')
plt.plot(x, u_lax_wendroff_results[-1, :], 'b-.', label='Лакс-Вендрофф')
plt.title("Сравнение с точным решением (t = t_max)")
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(x, error_explicit, 'r--', label='Ошибка Явная схема')
plt.plot(x, error_lax_wendroff, 'b-.', label='Ошибка Лакс-Вендрофф')
plt.title("Ошибка численных схем")
plt.xlabel('x')
plt.ylabel('Ошибка')
plt.legend()

plt.subplot(3, 2, 3)
for n in range(0, t_points, int(t_points / 10)):
    plt.plot(x, u_explicit_results[n, :], label=f't={t[n]:.2f}')
plt.title("Эволюция(Явная схема)")
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()

plt.subplot(3, 2, 4)
for n in range(0, t_points, int(t_points / 10)):
    plt.plot(x, u_lax_wendroff_results[n, :], label=f't={t[n]:.2f}')
plt.title("Эволюция(Лакс-Вендрофф)")
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()

plt.subplot(3, 2, 5)
for n in range(0, t_points, int(t_points / 10)):
    plt.plot(x, exact_solution(x, t[n]), label=f't={t[n]:.2f}')
plt.title("Эволюция точного")
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.legend()

plt.tight_layout()
plt.show()
