import cmath
import numpy
from matplotlib import pyplot as plt
import scipy


def f_transform(x, N, k):
    i = 0
    summ = 0
    while i < N:
        summ = summ + x[i] * cmath.exp(-(2 * cmath.pi * 1j * k * i / N))
        i = i + 1
    return summ


def generate_sine_wave(freq, sample_rate, duration):
    x = numpy.linspace(0, duration, sample_rate*duration, endpoint=False)
    frequencies = x * freq
    # 2pi для преобразования в радианы
    y = numpy.sin((2 * numpy.pi) * frequencies)
    return x, y


# Генерируем sin(x)
SAMPLE_RATE = 100  # Гц, сколько точек используется для представления синусоидальной волны на интервале 1 с
DURATION = 5  # Секунды

x, nice_sig = generate_sine_wave(1, SAMPLE_RATE, DURATION)
_, noise_sig = generate_sine_wave(5, SAMPLE_RATE, DURATION)

mixed_sig = nice_sig + noise_sig

N = SAMPLE_RATE * DURATION
mixed_sig_f = numpy.array(list(range(N)), dtype=numpy.complex64)
k = 0
while k < N:
    mixed_sig_f[k] = f_transform(mixed_sig, N, k)
    k = k+1
xf = x

fig, ax = plt.subplots()
ax.set_title('fft')
# отображение двух графиков в одной ск
#ax.plot(x, mixed_sig)
ax.plot(xf, numpy.abs(mixed_sig_f))

fig1, ax1 = plt.subplots()  # сигнал отдельно
ax1.set_title('Sin(x)')
ax1.plot(x, mixed_sig)

yf_table = scipy.fft.fft(mixed_sig)
ax.plot(xf, numpy.abs(yf_table))

fig1, ax2 = plt.subplots()
ax2.set_title('fft_difference')
y_difference = mixed_sig_f - yf_table
ax2.plot(xf, numpy.abs(y_difference))
# сетка
ax.grid()
# Подписи к осям:
ax.set_xlabel('Время, с')
ax.set_ylabel('Сигнал')

plt.show()
