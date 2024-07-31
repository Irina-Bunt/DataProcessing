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
    x = numpy.linspace(-duration, duration, sample_rate*duration, endpoint=False)
    frequencies = x * freq
    # 2pi для преобразования в радианы
    y = numpy.sin((2 * numpy.pi) * frequencies)
    return x, y


# Генерируем sin(x)
SAMPLE_RATE = 100  # Гц, сколько точек используется для представления синусоидальной волны на интервале 1 с
DURATION = 5  # Секунды
x, y = generate_sine_wave(2.13, SAMPLE_RATE, DURATION)  # частота 2Гц

# Берем ПФ от sin(x)
N = SAMPLE_RATE * DURATION
yf = numpy.array(list(range(N)), dtype=numpy.complex64)
yf_table = scipy.fft.fft(y)

k = 0
while k < N:
    yf[k] = f_transform(y, N, k)
    k = k+1
xf = x

fig, ax = plt.subplots()
ax.set_title('Sin(x)')
# отображение двух графиков в одной ск
ax.plot(x, y)

fig1, ax1 = plt.subplots()
ax1.set_title('fft')
ax1.plot(xf, numpy.abs(yf))

fig1, ax3 = plt.subplots()
ax3.set_title('fft_table')
ax3.plot(xf, numpy.abs(yf_table))

fig1, ax2 = plt.subplots()
ax2.set_title('fft_difference')
y_difference = yf - yf_table
ax2.plot(xf, numpy.abs(y_difference))
# сетка
ax.grid()
# Подписи к осям:
ax.set_xlabel('Время, с')
ax.set_ylabel('Сигнал')

plt.show()

