import numpy as np
import matplotlib.pyplot as plt

def bisect(func, a, b, precision=1e-6, max_iter=100):
    if func(a) * func(b) >= 0:
        raise ValueError("Функция должна иметь разные знаки на концах интервала [a, b].")

    for i in range(max_iter):
        mid = (a + b) / 2.0
        f_mid = func(mid)
        if abs(f_mid) < precision or (b - a) / 2.0 < precision:
            return mid

        if func(a) * f_mid < 0:
            b = mid
        else:
            a = mid
    return mid

def simple_iterations(func, x0, tol=1e-6, max_iter=100):
    x = x0
    for i in range(max_iter):
        x_new = func(x)
        if abs(x_new - x) < tol:
            return x_new

        x = x_new

    raise ValueError("Метод простых итераций не сошелся за максимальное количество итераций.")

def newton_method(func, dfunc, x0, tol=1e-6, max_iter=10000):
    x = x0
    for i in range(max_iter):
        fx = func(x)
        dfx = dfunc(x)
        if abs(dfx) < 1e-10:
            raise ValueError("Производная слишком близка к нулю.")
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new

        x = x_new

    raise ValueError("Метод Ньютона не сошелся за максимальное количество итераций.")

U0 = 10
a = 1

def f_even(E, m = 1):
    k = np.sqrt(2 * m * (U0 - E))
    kappa = np.sqrt(2 * m * E)
    return kappa - k * np.tan(k * a)

def f_odd(E, m = 1):
    k = np.sqrt(2 * m * (U0 - E))
    kappa = np.sqrt(2 * m * E)
    return kappa + k / np.tan(k * a)

def df_even(E, m = 1):
    k = np.sqrt(2 * m * (U0 - E))
    return m * ((a * k * (1 / np.cos(k * a))**2) + (U0 / (np.sqrt(E * (E + U0)) * k ** 2)))

def df_odd(E, m = 1):
    k = np.sqrt(2 * m * (U0 - E))
    kappa = np.sqrt(2 * m * E)
    return m * ((a * k * (1 / np.cos(k * a))**2) - (U0 / (np.sqrt(E * (E + U0)) * kappa ** 2)))


E_min = 0
E_max = U0


E_even_bisect = bisect(f_even, 9, 9.5)
E_odd_bisect = bisect(f_odd, 6, 7)

E_even_newton = newton_method(f_even, df_even, 0.9 * U0)
E_odd_newton = newton_method(f_odd, df_odd, 0.6 * U0)

def iteration_even(E, m = 1):
    k = np.sqrt(2 * m * (U0 - E))
    kappa = np.sqrt(2 * m * E)
    return U0 - ((np.arctan(kappa / k) / a) ** 2 / 2 * m)

def iteration_odd(E, m = 1):
    k = np.sqrt(2 * m * (U0 - E))
    kappa = np.sqrt(2 * m * E)
    return U0 - ((np.arctan(- k / kappa) / a) ** 2 / 2 * m)
    #return U0 - ((kappa / np.tan(k * a))) ** 2 / (2 * m)


E_even_simple = simple_iterations(iteration_even, 0.3 * U0)
E_odd_simple = simple_iterations(iteration_odd, 0.8 * U0)


print(E_even_bisect, E_even_newton, E_even_simple, E_odd_bisect, E_odd_newton, E_odd_simple)
E = np.linspace(0.01, 10.0, 400)


f_even_values = f_even(E)
f_odd_values = f_odd(E)
plt.figure(figsize=(10, 6))
plt.plot(E, f_even_values, label='f_even(E)', color='blue')
plt.plot(E, f_odd_values, label='f_odd(E)', color='red')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.xlabel('E (Энергия)')
plt.ylabel('f(E)')
plt.legend()
plt.grid(True)
plt.ylim(-10, 10)
plt.show()