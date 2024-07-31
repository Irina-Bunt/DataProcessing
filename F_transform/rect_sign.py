import cmath
import numpy
from matplotlib import pyplot as plt
import scipy
from scipy import signal
from statistics import mean


def f_transform(x, N, k):
    i = 0
    summ = 0
    while i < N:
        summ = summ + x[i] * cmath.exp(-(2 * cmath.pi * 1j * k * i / N))
        i = i + 1
    return summ


D = 5 # width of pulse, duration
sample_rate = 100
freq = 2
x = numpy.linspace(0, D*2, sample_rate*D, endpoint=False)
y_ = signal.square(2 * numpy.pi * freq * x) #+ 2*shift*numpy.pi)
#sig = numpy.sign(numpy.sin(2 * numpy.pi * freq * x))

y_avr = mean(y_)
print(y_avr)
j = 0
N1 = sample_rate*D
y = numpy.zeros(N1)
while j < N1:
    y[j] = y_[j] - y_avr
    j = j+1

# Берем ПФ
yf = numpy.array(list(range(N1)), dtype=numpy.complex64)
yf_table = scipy.fft.fft(y)

k = 0
while k < N1:
    yf[k] = f_transform(y, N1, k)
    k = k+1
xf = x

fig, ax = plt.subplots()
ax.set_title('Signal')
ax.plot(x, y)

fig1, ax1 = plt.subplots()
ax1.set_title('fft')
ax1.plot(xf, numpy.abs(yf))
ax1.plot(xf, numpy.abs(yf_table))

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

