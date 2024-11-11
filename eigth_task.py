import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

def f_explicit(t, u, v):
    """Функция для вычисления производных u и v."""
    du = 998 * u + 1998 * v
    dv = -999 * u - 1999 * v
    return du, dv

def explicit_euler(f, t0, u0, v0, t_end, h):
    """Явная схема Эйлера."""
    t_values = np.arange(t0, t_end + h, h)
    u_values = np.zeros_like(t_values)
    v_values = np.zeros_like(t_values)

    u_values[0], v_values[0] = u0, v0

    for i in range(1, len(t_values)):
        du, dv = f(t_values[i - 1], u_values[i - 1], v_values[i - 1])
        u_values[i] = u_values[i - 1] + h * du
        v_values[i] = v_values[i - 1] + h * dv

    return t_values, u_values, v_values

def trapezoidal_rule(f, t0, u0, v0, t_end, h):
    """схема трапеций."""
    t_values = np.arange(t0, t_end + h, h)
    u_values = np.zeros_like(t_values)
    v_values = np.zeros_like(t_values)

    u_values[0], v_values[0] = u0, v0

    for i in range(1, len(t_values)):
        du_old, dv_old = f(t_values[i - 1], u_values[i - 1], v_values[i - 1])
        u_guess = u_values[i - 1] + h * du_old
        v_guess = v_values[i - 1] + h * dv_old

        du_new, dv_new = f(t_values[i], u_guess, v_guess)
        u_values[i] = u_values[i - 1] + h / 2 * (du_old + du_new)
        v_values[i] = v_values[i - 1] + h / 2 * (dv_old + dv_new)

    return t_values, u_values, v_values

def implicit_euler(f, t0, u0, v0, t_end, h):
    """Неявная схема Эйлера."""
    t_values = np.arange(t0, t_end + h, h)
    u_values = np.zeros_like(t_values)
    v_values = np.zeros_like(t_values)

    u_values[0], v_values[0] = u0, v0

    A = np.array([[1 - h * 998, -h * 1998],
                  [h * 999, 1 + h * 1999]])

    for i in range(1, len(t_values)):
        u_old, v_old = u_values[i - 1], v_values[i - 1]
        b = np.array([u_old, v_old])
        u_new, v_new = solve(A, b)
        u_values[i], v_values[i] = u_new, v_new

    return t_values, u_values, v_values

t0 = 0
u0 = 1
v0 = 0
t_end = 0.05
h = 0.001

# t, u_explicit, v_explicit = explicit_euler(f_explicit, t0, u0, v0, t_end, h)
# _, u_trap, v_trap = trapezoidal_rule(f_explicit, t0, u0, v0, t_end, h)
# _, u_implicit, v_implicit = implicit_euler(f_explicit, t0, u0, v0, t_end, h)

def plot_phase_trajectories(u_values_list, v_values_list, labels):
    plt.figure(figsize=(10, 6))
    for u_values, v_values, label in zip(u_values_list, v_values_list, labels):
        plt.plot(u_values, v_values, label=label)
    plt.xlabel('u')
    plt.ylabel('v')
    plt.title('Фазовые траектории')
    plt.legend()
    plt.grid(True)
    plt.show()


# plot_phase_trajectories(
#     [u_explicit, u_trap, u_implicit],
#     [v_explicit, v_trap, v_implicit],
#     ['Явная схема Эйлера', 'Полунеявная схема трапеций', 'Неявная схема Эйлера']
# )

def plot_time_series(t_values_list, u_values_list, v_values_list, labels):
    plt.figure(figsize=(10, 6))
    for t_values, u_values, v_values, label in zip(t_values_list, u_values_list, v_values_list, labels):
        plt.plot(t_values, u_values, label=f'u ({label})', linestyle='--')
        plt.xlabel('t')
        plt.ylabel('u, v')
        plt.title('Численное решение')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.plot(t_values, v_values, label=f'v ({label})', linestyle='--')
        plt.xlabel('t')
        plt.ylabel('u, v')
        plt.title('Численное решение')
        plt.legend()
        plt.grid(True)
        plt.show()





step_sizes = [0.001, 0.005, 0.01, 0.06, 0.1]
# нарисовать для разных шагов по времени
for h in step_sizes:
    t, u_explicit, v_explicit = explicit_euler(f_explicit, t0, u0, v0, t_end, h)
    _, u_trap, v_trap = trapezoidal_rule(f_explicit, t0, u0, v0, t_end, h)
    _, u_implicit, v_implicit = implicit_euler(f_explicit, t0, u0, v0, t_end, h)

    plot_time_series(
        [t, t, t],
        [u_explicit, u_trap, u_implicit],
        [v_explicit, v_trap, v_implicit],
        [f'Явная схема {h}', f'Полунеявная схема {h}', f'Неявная схема {h}']
    )
