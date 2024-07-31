import cmath
import numpy
from matplotlib import pyplot as plt
import scipy
from scipy import signal
from statistics import mean
import random


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
nice_sig = signal.square(2 * numpy.pi * freq * x) #+ 2*shift*numpy.pi)
#y_ = numpy.sign(numpy.sin(2 * numpy.pi * freq * x))

y_avr = mean(nice_sig)
print(y_avr)
j = 0
N1 = sample_rate*D
y = numpy.zeros(N1)
while j < N1:
    y[j] = nice_sig[j] - y_avr
    j = j+1

noise_sig = numpy.zeros(N1)
k = 0
while k < N1:
    noise_sig[k] = random.random() * (1 + 1) - 1
    k = k+1

mixed_sig = nice_sig + noise_sig

mixed_sig_f = numpy.array(list(range(N1)), dtype=numpy.complex64)
k = 0
while k < N1:
    mixed_sig_f[k] = f_transform(mixed_sig, N1, k)
    k = k+1
xf = x

fig, ax = plt.subplots()
ax.set_title('fft')
# отображение двух графиков в одной ск
#ax.plot(x, mixed_sig)
ax.plot(xf, numpy.abs(mixed_sig_f))

fig1, ax1 = plt.subplots()  # сигнал отдельно
ax1.set_title('Rec_sign')
ax1.plot(x, mixed_sig)

yf_table = scipy.fft.fft(mixed_sig)
ax.plot(xf, numpy.abs(yf_table))

fig1, ax2 = plt.subplots()
ax2.set_title('difference')
y_difference = mixed_sig_f - yf_table
ax2.plot(xf, numpy.abs(y_difference))
# сетка
ax.grid()
# Подписи к осям:
ax.set_xlabel('Время, с')
ax.set_ylabel('Сигнал')

plt.show()

