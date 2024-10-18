import numpy as np
import matplotlib.pyplot as plt
#что такое жесктая схема и почему не сходится задача
def f(u, v):
    return 998 * u + 1998 * v, -999 * u - 1999 * v

def explicit_euler(u0, v0, t_end, h):
    steps = int(t_end / h) + 1
    u_values = np.zeros(steps)
    v_values = np.zeros(steps)
    u_values[0], v_values[0] = u0, v0

    for n in range(steps - 1):
        u_values[n + 1] = u_values[n] + h * f(u_values[n], v_values[n])[0]
        v_values[n + 1] = v_values[n] + h * f(u_values[n], v_values[n])[1]

    return u_values, v_values

def implicit_euler(u0, v0, t_end, h):
    steps = int(t_end / h) + 1
    u_values = np.zeros(steps)
    v_values = np.zeros(steps)
    u_values[0], v_values[0] = u0, v0

    for n in range(steps - 1):
        u_next = u_values[n] / (1 - h * 998) + h * v_values[n] / (1 - h * 1998)
        v_next = -u_values[n] / (1 - h * 999) - v_values[n] / (1 - h * 1999)
        u_values[n + 1] = u_next
        v_values[n + 1] = v_next

    return u_values, v_values

def runge_kutta(u0, v0, t_end, h):
    steps = int(t_end / h) + 1
    u_values = np.zeros(steps)
    v_values = np.zeros(steps)
    u_values[0], v_values[0] = u0, v0

    for n in range(steps - 1):
        k1u, k1v = f(u_values[n], v_values[n])
        k2u, k2v = f(u_values[n] + h * k1u, v_values[n] + h * k1v)
        u_values[n + 1] = u_values[n] + (h / 2) * (k1u + k2u)
        v_values[n + 1] = v_values[n] + (h / 2) * (k1v + k2v)

    return u_values, v_values

u0 = 1.0
v0 = 1.0
t_end = 0.01
h = 0.0001

u_explicit, v_explicit = explicit_euler(u0, v0, t_end, h)
u_implicit, v_implicit = implicit_euler(u0, v0, t_end, h)
u_rk, v_rk = runge_kutta(u0, v0, t_end, h)

plt.figure(figsize=(12, 8))
plt.plot(u_explicit, v_explicit, label='Явный метод Эйлера', color='blue')
plt.plot(u_implicit, v_implicit, label='Неявный метод Эйлера', color='red')
plt.plot(u_rk, v_rk, label='Метод Рунге-Кутты 2-го порядка', color='green')
plt.title('Сравнение решений жесткой системы уравнений')
plt.xlabel('u')
plt.ylabel('v')
plt.legend()
plt.grid()
plt.show()