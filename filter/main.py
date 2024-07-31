import numpy as np
from matplotlib import pyplot as plt
import random


def mov_avr(a, N):
    # a1 = np.concatenate((np.zeros(len(a)//4), a))
    M = len(a)
    numbers = [np.sum([(1/N * a[j-i]) for i in range(N)]) for j in range(24, M)]
    # print(nu)
    return numbers


def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate*duration, endpoint=False)
    frequencies = x * freq
    # 2pi для преобразования в радианы
    y = np.sin((2 * np.pi) * frequencies)
    return x, y


# Генерируем sin(x)
SAMPLE_RATE = 200  # Гц, сколько точек используется для представления синусоидальной волны на интервале 1 с
DURATION = 1  # Секунды
x, sine = generate_sine_wave(5, SAMPLE_RATE, DURATION)

N = SAMPLE_RATE * DURATION
noise_sig = np.zeros(N)
k = 0
while k < N:
    noise_sig[k] = random.random() * (1 + 0) - 0.5
    k = k+1

mixed_sig = sine + noise_sig
sig = mov_avr(mixed_sig, 25)

fig, ax = plt.subplots()
ax.set_title('Sine')
ax.plot(x, sine)

fig1, ax1 = plt.subplots()
ax1.set_title('Mov_avr')
ax1.plot(sig)

fig1, ax2 = plt.subplots()
ax2.set_title('mixed_sig')
ax2.plot(mixed_sig)
#
# fig1, ax2 = plt.subplots()
# ax2.set_title('Conv_fft')
# ax2.plot(ifft)

# сетка
ax.grid()
ax1.grid()
# Подписи к осям:
ax.set_xlabel('Время, с')
ax.set_ylabel('Сигнал')

plt.show()