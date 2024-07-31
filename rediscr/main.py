import numpy as np
import matplotlib.pylab as plt
import discr

def mat_draw_functions(x_line, *y_lines):
    for current_y_line in y_lines:
        plt.plot(x_line, current_y_line)
    plt.axis('tight')
    plt.show()


begin_conv_arg = -100
end_conv_arg = 100

fmax = 1000
T = 1.0 / fmax
T_points = 30
period_number = 10
N = (T_points * period_number + 1) * 1
x_line = np.linspace(0.0, T * N, 60)
signal = np.cos(2 * np.pi * x_line * 50) + np.cos(2 * np.pi * x_line * 150) + np.cos(2 * np.pi * x_line * 450)

# output_signal_spectrum = discr.cut_center_in_array(np.fft.fft(signal), relative_cutted_part_size=0.5)
# mat_draw_functions(np.linspace(0, 10, len(np.fft.fft(signal))), np.abs(np.fft.fft(signal)))
# mat_draw_functions(np.linspace(0, 10, len(output_signal_spectrum)), np.abs(output_signal_spectrum))
#
# output_signal_spectrum = discr.add_center_in_array(np.fft.fft(signal), relative_cutted_part_size=0.5)
# mat_draw_functions(np.linspace(0, 10, len(output_signal_spectrum)), np.abs(output_signal_spectrum))
#
#adding frequency
#------------------------------------------------------------------------
f_discr = 1 / (x_line[2] - x_line[1])
f_kot = (f_discr) / 2
f_line = np.linspace(0.0, fmax / 2, N // 2)
signal = np.cos(2 * np.pi * x_line * 10)
zeros_pad_array = discr.add_zeros(signal, 1)
# mat_draw_functions(np.linspace(0, 10, len(signal)), signal)
# mat_draw_functions(np.linspace(0, 10, len(zeros_pad_array)), zeros_pad_array)
spectrum = np.fft.fft(zeros_pad_array)
# mat_draw_functions(np.linspace(0, 10, len(spectrum)), spectrum)
parameter = 2
# spectrum = discr.cut_center_in_array(spectrum, relative_cutted_part_size=0.5, pass_zeros=True)
# mat_draw_functions(np.linspace(0, 10, len(spectrum)), spectrum)
_signal = discr.fill_average(signal, 1)
# mat_draw_functions(np.linspace(0, 10, len(_signal)), _signal, np.fft.ifft(spectrum) * 2)
# mat_draw_functions(np.linspace(0, 10, len(signal)), np.fft.fft(signal))
#------------------------------------------------------------------------

# signal = np.cos(2 * np.pi * x_line * 10)
# discr.discr_rel_number(signal, 2.25)
# spectrum = np.fft.fft(signal)
# spectrum = discr.cut_center_in_array(spectrum, relative_cutted_part_size=0.5, pass_zeros=True)
# # mat_draw_functions(np.linspace(0, 10, len(spectrum)), spectrum)
# output_signal = discr.take_every_step(np.fft.ifft(spectrum), 2)
# # mat_draw_functions(np.linspace(0, 10, len(signal)), discr.fill_average(output_signal, 1), signal)

signal = np.cos(2 * np.pi * x_line * 10)
coef = 0.75
output_signal = discr.discr_rel_number(signal, coef)
# mat_draw_functions(np.linspace(0, 10, len(output_signal)), output_signal, output_signal)
# mat_draw_functions(np.linspace(0, 10, len(signal)), signal, signal)


def new_x_line_discr(x_line, freq_coef):
    begin_x = x_line[0]
    end_x = x_line[-1]
    d = int(len(x_line) * freq_coef)
    x_list = [((end_x - begin_x) / (len(x_line) - 1)) * current_x_index / freq_coef for current_x_index in range(0, int(len(x_line) * freq_coef))]
    return np.asarray(x_list)

# метод целых передискретизаций
# plt.plot(x_line, signal, '.', label='50 точек')
# plt.plot(new_x_line_discr(x_line, coef), output_signal, '.', label='100 точек')
# plt.show()

#фурье метод


if coef < 1:
    output_signal_spectrum = discr.cut_center_in_array(np.fft.fft(signal), relative_cutted_part_size=1 - coef,
                                                       pass_zeros=False)
else:
    output_signal_spectrum = discr.add_center_in_array(np.fft.fft(signal), relative_cutted_part_size=coef - 1)
output_x = new_x_line_discr(x_line, coef)
plt.plot(x_line, signal, '.', label='исходный сигнал')
plt.plot(output_x, output_signal, '.', label='целая передискритезация')
plt.plot(output_x, np.real(np.fft.ifft(output_signal_spectrum))[:len(new_x_line_discr(x_line, coef))] * coef,
         '.', label='фурье')
plt.legend()
plt.show()