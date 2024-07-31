import numpy as np
from matplotlib import pyplot as plt
import random
import scipy
import cmath


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


def single_rec(x, right, width, ampl): #
    return ampl/2*(np.sign(x - right) * np.sign(-x + (right + width)) + 1)


x = np.linspace(0, 200, 200)
sigg1 = single_rec(x, 10, 10, 0.1)
sigg2 = single_rec(x, 10, 25, 0.04)
sigg3 = single_rec(x, 10, 50, 0.02)
am_fr_sig1 = scipy.fft.fft(sigg1)
am_fr_sig2 = scipy.fft.fft(sigg2)
am_fr_sig3 = scipy.fft.fft(sigg3)
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

fig1, ax3 = plt.subplots()
ax3.set_title('АЧХ')
ax3.set_xlim([0, 100])
ax3.plot(abs(am_fr_sig1))
ax3.plot(abs(am_fr_sig2))
ax3.plot(abs(am_fr_sig3))

fig1, ax3 = plt.subplots()
ax3.set_title('ФЧХ')
ax3.set_xlim([0, 50])
ax3.plot([cmath.phase(am_fr_sig1[i]) for i in range(200)])
ax3.plot([cmath.phase(am_fr_sig2[i]) for i in range(200)])
ax3.plot([cmath.phase(am_fr_sig3[i]) for i in range(200)])

# сетка
ax.grid()
ax1.grid()
# Подписи к осям:
ax.set_xlabel('Время, с')
ax.set_ylabel('Сигнал')

plt.show()