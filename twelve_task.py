import numpy as np
import matplotlib.pyplot as plt

def DFT_slow(x, freq, dt):
    x = np.asarray(x, dtype=float)
    n = np.arange(len(x))
    exponent = -2j * np.pi * freq[:, np.newaxis] * n * dt
    exp_matrix = np.exp(exponent)
    X = np.dot(exp_matrix, x)
    return X

a0 = 1
a1 = 0.002
omega0 = 5.1
omega1 = 25.5
T = 2 * np.pi

N = 1024
t = np.linspace(0, T, N, endpoint=False)
dt = t[1] - t[0]
fs = 1 / dt

f = a0 * np.sin(omega0 * t) + a1 * np.sin(omega1 * t)
w = len(f) / 2
window_rect = np.ones_like(f)
window_rect[:int(len(f) / 2 - w / 2)] = 0
window_rect[int(len(f) / 2 + w / 2):] = 0
f_rect = f * window_rect

window_hann = np.hanning(N)
f_hann = f * window_hann

# Создаем единую ось частот
freq = np.fft.fftfreq(N, d=dt)
freq_rad = freq * 2 * np.pi
freq_rad_shifted = np.fft.fftshift(freq_rad)

# Вычисляем DFT с использованием единой оси частот
F_rect = DFT_slow(f_rect, freq, dt)
F_rect_shifted = np.fft.fftshift(F_rect)
power_spectrum_rect = np.abs(F_rect_shifted) ** 2
power_spectrum_rect /= np.max(power_spectrum_rect)

F_hann = DFT_slow(f_hann, freq, dt)
F_hann_shifted = np.fft.fftshift(F_hann)
power_spectrum_hann = np.abs(F_hann_shifted) ** 2
power_spectrum_hann /= np.max(power_spectrum_hann)

# Построение графиков с использованием единой оси частот
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(freq_rad_shifted, power_spectrum_rect)
plt.title('С прямоугольным окном')
plt.xlabel('Частота (рад/с)')
plt.ylabel('Нормированная мощность')
plt.xlim(-40, 40)
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(freq_rad_shifted, power_spectrum_hann)
plt.title('С окном Ханна')
plt.xlabel('Частота (рад/с)')
plt.ylabel('Нормированная мощность')
plt.xlim(-40, 40)
plt.grid(True)

plt.tight_layout()
plt.show()
