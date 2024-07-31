import numpy as np
from matplotlib import pyplot as plt
import random
import scipy


def generate_cosine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate*duration, endpoint=False)
    frequencies = x * freq
    # 2pi для преобразования в радианы
    y = np.cos((2 * np.pi) * frequencies)
    return x, y


def single_rec(x, right, width, ampl): #
    return ampl/2*(np.sign(x - right) * np.sign(-x + (right + width)) + 1)

# ФНЧ
x = np.linspace(0, 100, 1300)
sig_freq = single_rec(x, -1, 2.5, 1) # частота среза ~32.5
sig_phas = np.zeros(len(x))

sig1 = sig_freq[:650]
sigg = single_rec(x, 97, 2.5, 1)
sig2 = sigg[650:]
sig3 = np.concatenate((sig1, sig2))

spectre_lpf = sig3 * (np.cos(sig_phas)+np.sin(sig_phas)*1j)
result = scipy.fft.ifft(spectre_lpf)

# ФВЧ
x = np.linspace(0, 100, 1300)
sig3 = single_rec(x, 5, 90, 1) # частота среза ~65
sig_phas = np.zeros(len(x))
# sig1 = sig_freq[:650]
# sigg = single_rec(x, 99, 2, 1)
# sig2 = sigg[650:]
# sig3 = np.concatenate((sig1, sig2))
spectre_hpf = sig3 * (np.cos(sig_phas)+np.sin(sig_phas)*1j)
result = scipy.fft.ifft(spectre_hpf)

# Генерируем sin(x) для свёртки с res_cut2
SAMPLE_RATE = 1300  # Гц, сколько точек используется для представления синусоидальной волны на интервале 1 с
DURATION = 1  # Секунды
_, cos1 = generate_cosine_wave(10, SAMPLE_RATE, DURATION)
_, cos2 = generate_cosine_wave(50, SAMPLE_RATE, DURATION)
_, cos3 = generate_cosine_wave(100, SAMPLE_RATE, DURATION)
noise_sig = np.random.normal(0, 1, 1300)
mixed_sig = cos1 + cos2 + cos3 + noise_sig
# mixed_sig2 = cos1


# ФНЧ + ФВЧ
band_pass = spectre_lpf + spectre_hpf
filtered = np.real(mixed_sig - np.fft.ifft(band_pass * np.fft.fft(mixed_sig)))

# mul = np.real(np.fft.ifft(np.fft.fft(mixed_sig) * sig))
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

fig, ax3 = plt.subplots()
ax3.set_title('Signal')
ax3.plot(x, band_pass)

fig, ax4 = plt.subplots(2, 1, sharex=True)
ax4[0].set_title('orig / Band-pass')
ax4[0].plot(x, mixed_sig)
ax4[1].plot(x, filtered)

fig, ax5 = plt.subplots()
ax5.set_title('mix_cosine')
ax5.set_xlim([0, 50])
ax5.plot(x, mixed_sig)

fig, ax6 = plt.subplots()
ax6.set_title('АЧХ_mix_sig')
ax6.set_xlim([0, 150])
ax6.plot(abs(scipy.fft.fft(mixed_sig)))
ax6.plot(abs(scipy.fft.fft(filtered)))

# fig, ax7 = plt.subplots()
# ax7.set_title('АЧХ_filter')
# ax7.set_xlim([0, 150])
# # ax7.plot(abs(scipy.fft.fft(res_cut2)))
# ax7.plot(abs(scipy.fft.fft(mul)))

plt.show()

