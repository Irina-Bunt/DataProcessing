def resampling(x, signal, M):
    def upSample(N, M, spectrum):
        newSpectrum = np.zeros(M, dtype=np.complex128)
        if (N % 2 == 0):
            for i in range(0, (N + 1) // 2 - 1):
                newSpectrum[i] = spectrum[i]
            for i in range((N + 1) // 2, (N + 1) // 2 + M - N - 1):
                newSpectrum[i] = 0
            for i in range((N + 1) // 2 + M - N, M - 1):
                newSpectrum[i] = spectrum[i - M + N]
        else:
            for i in range(0, N // 2 - 1):
                newSpectrum[i] = spectrum[i]
            newSpectrum[N // 2] = spectrum[(N // 2)] // 2
            for i in range(N // 2 + 1, N // 2 + M - N - 1):
                newSpectrum[i] = 0
            newSpectrum[N // 2 + M - M] = spectrum[(N // 2)] // 2
            for i in range(N // 2 + M - N + 1, M - 1):
                newSpectrum[i] = spectrum[i - M + N]
        newSpectrum = newSpectrum * M / N
        return newSpectrum

    N = len(signal)
    spectrum = np.fft.fft(signal)
    newX = np.arange(x[0], x[len(x) - 1], (x[len(x) - 1] - x[0]) / M)[:M]
    tmpM = N * M // math.gcd(N, M)
    print(tmpM)
    newSpectrumTMP = upSample(N, tmpM, spectrum)
    newSignalTMP = np.fft.ifft([newSpectrumTMP[i] if (i < M / 2 or i > (tmpM - M / 2)) else 0 for i in range(tmpM)])
    # newSignalTMP = np.fft.ifft(newSpectrumTMP)
    newSignal = np.zeros(M)
    for i in range(M):
        newSignal[i] = newSignalTMP[int(i / M * tmpM)]
    return newX, np.real(newSignal)


def freq(x: np.array):
    return np.fft.fftfreq(x.size, d=x[1] - x[0])


plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100
sampling = 0.01
x = np.arange(-xRange, xRange, sampling)
fg, ax = plt.subplots(2, 2)
my_signal = signal.square(2 * np.pi * x)
ax[0, 0].plot(x, my_signal)
ax[0, 1].plot(np.abs(np.fft.fft(my_signal)))
new_x, resampled = resampling(x, my_signal, 1500)
ax[1, 0].plot(new_x, resampled)
ax[1, 1].plot(np.abs(np.fft.fft(resampled)))