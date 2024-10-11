import numpy as np
import matplotlib.pyplot as plt

def system(t, u, v):
    a = 10
    b = 2
    c = 2
    d = 10
    return a * u - b * u * v, c * u * v - d * v

def rk2_solver(model, start_t, start_u, start_v, stop_t, step_size):
    time = np.arange(start_t, stop_t, step_size)
    prey = np.zeros_like(time)
    pred = np.zeros_like(time)
    prey[0], pred[0] = start_u, start_v

    for idx in range(1, len(time)):
        t_prev = time[idx - 1]
        u_prev, v_prev = prey[idx - 1], pred[idx - 1]
        k1_u, k1_v = model(t_prev, u_prev, v_prev)
        k2_u, k2_v = model(t_prev + step_size, u_prev + step_size * k1_u, v_prev + step_size * k1_v)
        prey[idx] = u_prev + step_size * (k1_u + k2_u) / 2
        pred[idx] = v_prev + step_size * (k1_v + k2_v) / 2

    return time, prey, pred


start_time = 0
prey_initial = 40
pred_initial = 9
end_time = 30
h = 0.01
time_steps, prey_population, predator_population = rk2_solver(system, start_time, prey_initial, pred_initial, end_time, h)

plt.figure(figsize=(8, 6))
plt.plot(prey_population, predator_population, color='red', label='Prey vs Predator')
plt.xlabel('Number Pray')
plt.ylabel('Number Predator')
plt.grid(True)
plt.legend()
plt.show()
