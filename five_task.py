import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.signal import argrelextrema

def lagrange_interpolation(x_nodes, y_nodes, x_values):
    n = len(x_nodes)
    P_n = np.zeros_like(x_values)

    for i in range(n):
        l_i = np.ones_like(x_values)
        for j in range(n):
            if i != j:
                l_i *= (x_values - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        P_n += y_nodes[i] * l_i

    return P_n

def cos(x):
    return np.cos(x)

def plot_interpolation_error(n_values, x_range, num_maxima=3):
    x_values = np.linspace(x_range[0], x_range[1], 1000)

    plt.figure(figsize=(10, 6))

    for n in n_values:
        x_nodes = np.linspace(x_range[0], x_range[1], n)
        y_nodes = cos(x_nodes)
        P_n = lagrange_interpolation(x_nodes, y_nodes, x_values)

        error = P_n - cos(x_values)
        plt.plot(x_values, error, label=f'n = {n}')

        #display max
        maxima_indices = argrelextrema(error, np.greater)[0]
        minima_indices = argrelextrema(error, np.less)[0]
        extrema_indices = np.concatenate((maxima_indices, minima_indices))
        extrema_x = x_values[extrema_indices]
        extrema_y = error[extrema_indices]

        if len(extrema_y) > num_maxima:
            sorted_indices = np.argsort(np.abs(extrema_y))[-num_maxima:]
            extrema_x = extrema_x[sorted_indices]
            extrema_y = extrema_y[sorted_indices]

        plt.scatter(extrema_x, extrema_y, color='red', zorder=5)
        for x_ext, y_ext in zip(extrema_x, extrema_y):
            distance_to_zero = np.abs(x_ext)
            plt.annotate(f'{distance_to_zero:.2f}', (x_ext, y_ext), textcoords="offset points", xytext=(0, 10), ha='center')

    plt.title("Interpolation error P_n(x) - cos(x)")
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.show()

# Задаем параметры
n_values = [5, 10, 20, 30]
x_range = [0, 10]
plot_interpolation_error(n_values, x_range)
