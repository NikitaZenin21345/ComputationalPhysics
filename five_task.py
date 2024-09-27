import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv


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

def J0(x):
    return jv(0, x)

def cos(x):
    return 1 / (x**2 + 1)

def plot_interpolation_error(n_values, x_range):
    x_values = np.linspace(x_range[0], x_range[1], 1000)

    plt.figure(figsize=(10, 6))

    for n in n_values:

        x_nodes = np.linspace(x_range[0], x_range[1], n)
        y_nodes = J0(x_nodes)
        P_n = lagrange_interpolation(x_nodes, y_nodes, x_values)

        error = P_n - J0(x_values)
        plt.plot(x_values, error, label=f'n = {n}')
        #print(f"{n} - {error}")

    plt.title("Interpolation error P_n(x) - J_0(x)")
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.show()

# 1 / (x^2 + 1)
n_values = [5, 10, 20, 30]
x_range = [0, 10]
plot_interpolation_error(n_values, x_range)


