import numpy as np
from matplotlib import pyplot as plt
import random
import scipy


def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate*duration, endpoint=False)
    frequencies = x * freq
    # 2pi для преобразования в радианы
    y = np.sin((2 * np.pi) * frequencies)
    return x, y


def single_rec(x, right, width, ampl): #
    return ampl/2*(np.sign(x - right) * np.sign(-x + (right + width)) + 1)

x = np.linspace(0, 100, 1300)
# freq_discr = 1/(x[2] - x[1])
# freq_kot = freq_discr/2
sig3 = single_rec(x, 5, 90, 1)
sig_phas = np.zeros(len(x))

# sig1 = sig_freq[:650]
# sigg = single_rec(x, 99, 2, 1)
# sig2 = sigg[650:]
# sig3 = np.concatenate((sig1, sig2))

sig = sig3 * (np.cos(sig_phas)+np.sin(sig_phas)*1j)
result = scipy.fft.ifft(sig)

res_cut1 = np.concatenate((result[:100], np.zeros(1100)))
res_cut2 = np.concatenate((res_cut1, result[1200:]))

# Генерируем sin(x) для свёртки с res_cut2
SAMPLE_RATE = 1300  # Гц, сколько точек используется для представления синусоидальной волны на интервале 1 с
DURATION = 1  # Секунды
_, sine1 = generate_sine_wave(80, SAMPLE_RATE, DURATION)
_, sine2 = generate_sine_wave(35, SAMPLE_RATE, DURATION)
noise_sig = np.random.normal(0, 1, 1300)
mixed_sig = sine1 + sine2 + noise_sig


mul = np.real(np.fft.ifft(np.fft.fft(mixed_sig) * sig))

# conv_table = np.convolve(res_cut2, mixed_sig)
x_new = np.linspace(0, 100, 2599)

# fig, ax1 = plt.subplots()
# ax1.set_title('Ampl_Phas')
# ax1.set_xlim([0, 100])
# ax1.plot(x, sig_phas)
# ax1.plot(x, sig3)
#
# fig, ax2 = plt.subplots()
# ax2.set_title('Signal')
# ax2.set_xlim([0, 100])
# ax2.plot(x, result)
#
# fig, ax3 = plt.subplots()
# ax3.set_title('Signal_cut')
# ax3.plot(x, res_cut2)

fig, ax4 = plt.subplots()
ax4.set_title('Conv_cut_noise')
ax4.set_xlim([0, 25])
ax4.plot(x, mul)

fig, ax5 = plt.subplots()
ax5.set_title('mix_sine')
ax5.set_xlim([0, 50])
ax5.plot(x, mixed_sig)

fig, ax6 = plt.subplots()
ax6.set_title('АЧХ_mix_sig')
ax6.set_xlim([0, 600])
ax6.plot(abs(scipy.fft.fft(mixed_sig)))
ax6.plot(abs(scipy.fft.fft(mul)))

fig, ax7 = plt.subplots()
ax7.set_title('АЧХ_filter')
ax7.set_xlim([0, 600])
ax7.plot(abs(scipy.fft.fft(res_cut2)))

plt.show()

