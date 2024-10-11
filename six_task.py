import numpy as np
import matplotlib.pyplot as plt

def equation_rhs(time, y_value):
    return -y_value


def euler_solver(rhs_func, t_start, y_init, t_final, step_size):
    time_steps = np.arange(t_start, t_final + step_size, step_size)
    y_approx = np.zeros_like(time_steps)
    y_approx[0] = y_init
    for idx in range(1, len(time_steps)):
        y_approx[idx] = y_approx[idx - 1] + step_size * rhs_func(time_steps[idx - 1], y_approx[idx - 1])
    return time_steps, y_approx


def rk2_solver(rhs_func, t_start, y_init, t_final, step_size):
    time_steps = np.arange(t_start, t_final + step_size, step_size)
    y_approx = np.zeros_like(time_steps)
    y_approx[0] = y_init
    for idx in range(1, len(time_steps)):
        slope1 = rhs_func(time_steps[idx - 1], y_approx[idx - 1])
        slope2 = rhs_func(time_steps[idx - 1] + step_size, y_approx[idx - 1] + step_size * slope1)
        y_approx[idx] = y_approx[idx - 1] + step_size * (slope1 + slope2) / 2
    return time_steps, y_approx


def rk4_solver(rhs_func, t_start, y_init, t_final, step_size):
    time_steps = np.arange(t_start, t_final + step_size, step_size)
    y_approx = np.zeros_like(time_steps)
    y_approx[0] = y_init
    for idx in range(1, len(time_steps)):
        k1 = rhs_func(time_steps[idx - 1], y_approx[idx - 1])
        k2 = rhs_func(time_steps[idx - 1] + step_size / 2, y_approx[idx - 1] + (step_size / 2) * k1)
        k3 = rhs_func(time_steps[idx - 1] + step_size / 2, y_approx[idx - 1] + (step_size / 2) * k2)
        k4 = rhs_func(time_steps[idx - 1] + step_size, y_approx[idx - 1] + step_size * k3)
        y_approx[idx] = y_approx[idx - 1] + (step_size / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return time_steps, y_approx


def analytical_solution(t_values):
    return np.exp(-t_values)


start_time = 0
initial_value = 1
end_time = 3

grid_points = [50, 100, 1000]

euler_errors = []
rk2_errors = []
rk4_errors = []

for points in grid_points:
    step = (end_time - start_time) / points

    t_euler, y_euler = euler_solver(equation_rhs, start_time, initial_value, end_time, step)
    t_rk2, y_rk2 = rk2_solver(equation_rhs, start_time, initial_value, end_time, step)
    t_rk4, y_rk4 = rk4_solver(equation_rhs, start_time, initial_value, end_time, step)

    exact_euler = analytical_solution(t_euler)
    exact_rk2 = analytical_solution(t_rk2)
    exact_rk4 = analytical_solution(t_rk4)

    euler_err = np.abs(exact_euler - y_euler)
    rk2_err = np.abs(exact_rk2 - y_rk2)
    rk4_err = np.abs(exact_rk4 - y_rk4)

    euler_errors.append(np.max(euler_err))
    rk2_errors.append(np.max(rk2_err))
    rk4_errors.append(np.max(rk4_err))

plt.figure(figsize=(10, 6))

plt.plot(grid_points, euler_errors, 'o-', label='Euler Method')
plt.plot(grid_points, rk2_errors, 's-', label='Runge-Kutta 2nd Order')
plt.plot(grid_points, rk4_errors, 'x-', label='Runge-Kutta 4th Order')

plt.xlabel('Number of Points')
plt.ylabel('Maximum Error')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()
