import numpy as np
import matplotlib.pyplot as plt


def f_u(u, v):
    return 998 * u + 1998 * v

def f_v(u, v):
    return -999 * u - 1999 * v


u0 = -3
v0 = 5


def exact_solution(points, t_end):
    time = np.linspace(0, t_end, points + 1)
    u = 2 * (u0 + v0) * np.exp(-time) + (-u0 - 2 * v0) * np.exp(-1000 * time)
    v = - (u0 + v0) * np.exp(-time) - (-u0 - 2 * v0) * np.exp(-1000 * time)
    return time, u, v


def explicit_euler(u0, v0, time):
    points = len(time) - 1
    h = time[1] - time[0]
    u = np.zeros(points + 1)
    v = np.zeros(points + 1)
    u[0], v[0] = u0, v0
    for n in range(points):
        u[n + 1] = u[n] + h * (998 * u[n] + 1998 * v[n])
        v[n + 1] = v[n] + h * (-999 * u[n] - 1999 * v[n])
    return u, v


def implicit_euler(u0, v0, time):
    points = len(time) - 1
    h = time[1] - time[0]
    u = np.zeros(points + 1)
    v = np.zeros(points + 1)
    u[0], v[0] = u0, v0
    for n in range(points):
        A = np.array([[1 - h * 998, -h * 1998], [h * 999, 1 + h * 1999]])
        b = np.array([u[n], v[n]])
        u_next, v_next = np.linalg.solve(A, b)
        u[n + 1], v[n + 1] = u_next, v_next
    return u, v


def runge_kutta(u0, v0, time):
    points = len(time) - 1
    h = time[1] - time[0]
    u = np.zeros(points + 1)
    v = np.zeros(points + 1)
    u[0], v[0] = u0, v0
    for n in range(points):
        u_star = u[n] + h * f_u(u[n], v[n])
        v_star = v[n] + h * f_v(u[n], v[n])
        u[n + 1] = u[n] + h / 2 * (f_u(u[n], v[n]) + f_u(u_star, v_star))
        v[n + 1] = v[n] + h / 2 * (f_v(u[n], v[n]) + f_v(u_star, v_star))
    return u, v


t_end = 1
grid_points = [1000, 10000, 100000, 1000000]

error_explicit = []
error_implicit = []
error_rk = []

for points in grid_points:
    time, u_analytical, v_analytical = exact_solution(points, t_end)

    u_values1, v_values1 = explicit_euler(u0, v0, time)
    u_values2, v_values2 = implicit_euler(u0, v0, time)
    u_values3, v_values3 = runge_kutta(u0, v0, time)

    error_explicit_u = np.abs(u_analytical - u_values1)
    error_implicit_u = np.abs(u_analytical - u_values2)
    error_rk_u = np.abs(u_analytical - u_values3)

    error_explicit.append(np.max(error_explicit_u))
    error_implicit.append(np.max(error_implicit_u))
    error_rk.append(np.max(error_rk_u))

    plt.figure(figsize=(10, 6))
    plt.plot(time, u_analytical, ':', label='Точное решение u', color='blue')
    plt.plot(time, u_values1, '-', label='Явный метод Эйлера u', color='red')
    plt.plot(time, u_values2, '--', label='Неявный метод Эйлера u', color='purple')
    plt.plot(time, u_values3, '-.', label='Полунеявный метод u', color='cyan')
    plt.xlabel('t')
    plt.ylabel('Значение функции')
    plt.title(f'Решения методов при {points} точках')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


plt.figure(figsize=(10, 6))
plt.plot(grid_points, error_explicit, label='Явный метод Эйлера', color='blue')
plt.plot(grid_points, error_implicit, label='Неявный метод Эйлера', color='red')
plt.plot(grid_points, error_rk, label='Полунеявный метод', color='green')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Количество шагов (по логарифмической шкале)')
plt.ylabel('Ошибка (по логарифмической шкале)')
plt.title('Ошибка численных методов относительно аналитического решения')
plt.legend()
plt.grid(True)
plt.show()