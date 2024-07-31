import numpy as np
from matplotlib import pyplot as plt
import scipy
import random


def convolution(a, b, M, N):
    cycle = N + M - 1
    a1 = np.concatenate((a, np.zeros(cycle - M)))
    b1 = np.concatenate((b, np.zeros(cycle - N)))
    numbers = [np.sum([(a1[i] * b1[j-i]) for i in range(M)]) for j in range(cycle)]
    return numbers


def rect_wave(x, c, c0):  # Прямоугольная волна с начальной точкой c0 и шириной c
    if x >= (c + c0):
        r = 0.0
    elif x < c0:
        r = 0.0
    else:
        r = 1
    return r


x = np.linspace(-2, 4, 500)
# y1 = np.array([rect_wave(t, 2, 0) for t in x])
# y2 = np.array([rect_wave(t, 2, -1) for t in x])

N = 500
y1 = np.zeros(N)
y2 = np.zeros(N)
k = 0
while k < N:
    y1[k] = random.random() * (1 + 1) - 1
    y2[k] = random.random() * (1 + 1) - 1
    k = k+1
y2 = y1.copy()
M = 500
conv = convolution(y1, y2, M, N)

conv_table = np.convolve(y1, y2)

cycle = N + M - 1
y1_ = np.concatenate((y1, np.zeros(cycle - M)))
y2_ = np.concatenate((y2, np.zeros(cycle - N)))
fft_y1 = scipy.fft.fft(y1_)
fft_y2 = scipy.fft.fft(y2_)

ifft = scipy.fft.ifft(fft_y1 * fft_y2)

fig, ax = plt.subplots()
ax.set_title('Rec_sign')
ax.plot(x, y1)
ax.plot(x, y2)

fig1, ax1 = plt.subplots()
ax1.set_title('Conv_sig')
ax1.plot(conv)

fig1, ax2 = plt.subplots()
ax2.set_title('Conv_sig_t')
ax2.plot(conv_table)

fig1, ax2 = plt.subplots()
ax2.set_title('Conv_fft')
ax2.plot(ifft)

fig1, ax2 = plt.subplots()
ax2.set_title('difference')
difference1 = conv - conv_table
difference2 = conv - ifft
ax2.plot(difference1)
ax2.plot(difference2)
# сетка
ax.grid()
# Подписи к осям:
ax.set_xlabel('Время, с')
ax.set_ylabel('Сигнал')

plt.show()