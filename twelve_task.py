import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import windows

def get_freq_array(length, spacing_rate):
    results = np.zeros(length, int)
    N = (length - 1) // 2 + 1
    results[:N] = np.arange(0, N, dtype=int)
    results[N:] = np.arange(-(length // 2), 0, dtype=int)
    return results / (length * spacing_rate)

def DFT_slow(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

a0 = 1
a1 = 0.002
w0 = 5.1
w1 = 5 * w0
T = 2 * np.pi
N = 1000
t = np.linspace(0, T, N)

f_t = a0 * np.sin(w0 * t) + a1 * np.sin(w1 * t)

rect_window = np.ones(N)

hanning_window = windows.hann(N)

spectrum_rect = DFT_slow(f_t)
frequencies = get_freq_array(N, 1 / N)

spectrum_hanning = DFT_slow(f_t * hanning_window)

power_spectrum_rect = np.abs(spectrum_rect)
power_spectrum_hanning = np.abs(spectrum_hanning)

mask = frequencies > 0

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(frequencies[mask], power_spectrum_rect[mask] / (T), label='Прямоугольное окно', color='blue')
plt.title('Прямоугольное окно')
plt.xlabel('Частота [Гц]')
plt.ylabel('Мощность')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(frequencies[mask], power_spectrum_hanning[mask] / (T), label='Окно Ханна', color='green')
plt.title('Окно Ханна')
plt.xlabel('Частота [Гц]')
plt.ylabel('Мощность')
plt.grid(True)

plt.tight_layout()
plt.show()
