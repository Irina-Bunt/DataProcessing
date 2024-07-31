import numpy as np
from matplotlib import pyplot as plt
import scipy


def correlation(a, b, M, N):
    cycle = N + M - 1
    a1 = np.concatenate((a, np.zeros(cycle - M)))
    b1 = np.concatenate((np.zeros(cycle - N), b))
    # corr_table = np.correlate(a1, b1, 'full')
    numbers = [np.sum([(a1[i] * b1[(j+i) % cycle]) for i in range(M)]) for j in range(cycle)]
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

x, sine1 = generate_sine_wave(5, SAMPLE_RATE, DURATION)
_, sine2 = generate_sine_wave(5, 300, 1)

M = 200
N = 300
corr = correlation(sine1, sine2, M, N)

cycle = N + M - 1
y1_ = np.concatenate((sine1, np.zeros(cycle - M)))
y2_ = np.concatenate((sine2, np.zeros(cycle - N)))
# corr_table = np.correlate(y1_, y2_, 'full')

fft_y1 = scipy.fft.fft(y1_)
fft_y2 = scipy.fft.fft(y2_)
ifft = scipy.fft.ifft(fft_y1 * fft_y2)

fig, ax = plt.subplots()
ax.set_title('Sine')
ax.plot(sine1)
ax.plot(sine2)

fig1, ax1 = plt.subplots()
ax1.set_title('Corr_sig')
ax1.plot(corr)

# fig1, ax2 = plt.subplots()
# ax2.set_title('Corr_sig_t')
# ax2.plot(corr_table)

fig1, ax2 = plt.subplots()
ax2.set_title('Corr_fft')
ax2.plot(ifft)

# сетка
ax.grid()
# Подписи к осям:
ax.set_xlabel('Время, с')
ax.set_ylabel('Сигнал')

plt.show()

