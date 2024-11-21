import numpy as np
import matplotlib.pyplot as plt

N = 1000  #
x_max = 5
x_min = -x_max
x = np.linspace(x_min, x_max, N + 2)
dx = x[1] - x[0]
x_internal = x[1:-1]
U = 0.5 * x_internal ** 2

diagonal = 1 / dx ** 2 + U
off_diagonal = -0.5 / dx ** 2 * np.ones(N - 1)

H = np.diag(diagonal) + np.diag(off_diagonal, k=1) + np.diag(off_diagonal, k=-1)

psi = np.random.rand(N)
norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
psi /= norm

E_old = 0

for i in range(1000):
    psi_new = np.linalg.solve(H, psi)

    norm = np.sqrt(np.sum(np.abs(psi_new) ** 2) * dx)
    psi_new /= norm

    E_new = np.dot(psi_new, H @ psi_new) * dx

    if np.abs(E_new - E_old) < 1e-10:
        print(i)
        break

    psi = psi_new
    E_old = E_new

psi_num = psi_new
E_num = E_new
psi_full = np.zeros(N + 2)
psi_full[1:-1] = psi_num  #

psi_analytical_full = (1 / np.pi) ** 0.25 * np.exp(-x ** 2 / 2)

norm_analytical = np.sqrt(np.sum(np.abs(psi_analytical_full) ** 2) * dx)
psi_analytical_full /= norm_analytical

plt.plot(x, psi_full - psi_analytical_full, label='Численное ψ(x)')
plt.xlabel('x')
plt.ylabel('ψ(x)')
plt.legend()
plt.show()

print("E0 =", E_num)
print("E =", 0.5)


