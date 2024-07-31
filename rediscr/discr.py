import numpy as np
from matplotlib import pyplot as plt

import math

def cut_center_in_array(array, relative_cutted_part_size, pass_zeros=False):
    cut_index = math.ceil((len(array) * relative_cutted_part_size) // 2)
    begin_cut_index = (len(array) // 2) - cut_index
    end_cut_index = (len(array) // 2) + cut_index
    if pass_zeros:
        return np.concatenate((array[:begin_cut_index],
                               np.zeros(end_cut_index - begin_cut_index), array[end_cut_index:]))
    else:
        return np.concatenate((array[:begin_cut_index], array[end_cut_index:]))


def add_center_in_array(array, relative_cutted_part_size):
    zeros_len = int(len(array) * relative_cutted_part_size)
    return np.concatenate((array[:len(array) // 2], np.zeros(zeros_len), array[len(array) // 2:]))


def add_zeros(array, zeros_per_point_count):
    output_array = np.zeros(len(array) * (zeros_per_point_count + 1))
    for current_index in range(len(array)):
        output_array[current_index * (zeros_per_point_count + 1)] = array[current_index]

    return output_array


def fill_average(signal, add_per_point):
    output_array = np.zeros(len(signal) * (add_per_point + 1))
    for current_index in range(len(signal) - 1):
        factor = (signal[current_index + 1] - signal[current_index]) / (add_per_point + 1)
        for zeros_index in range(add_per_point + 1):
            output_array[(current_index * (add_per_point + 1)) + zeros_index] = signal[current_index] + factor * zeros_index

    return output_array


def take_every_step(signal, step):
    output = np.zeros(int(len(signal) // step))
    j = 0
    for i in range(0, len(signal) - step, step):
        output[j] = np.real(signal[i])
        j += 1
    return output


def get_boundary_frequency_index(frequency_line, boundary_frequency):
    for current_index, current_frequency in enumerate(frequency_line):
        if current_frequency > boundary_frequency:
            return current_index
    raise


def sym_reflect(signal, f_index, change_sign=True):
    first_half = signal[:f_index]
    second_half = np.copy(np.flip(first_half))
    if change_sign:
        second_half *= -1
    return np.concatenate((first_half, second_half[:-1]))


def mat_draw_functions(x_line, *y_lines):
    for current_y_line in y_lines:
        plt.plot(x_line, current_y_line)
    plt.axis('tight')
    plt.show()


def get_perfect_filter_spectrum(frequency_line, boundary_frequency, f_kot, high_manager):
    f_kot_index = get_boundary_frequency_index(frequency_line, f_kot)
    # boundary_index = get_boundary_frequency_index(frequency_line, boundary_frequency)
    ampl_filter = high_manager(frequency_line, boundary_frequency)
    ampl_filter = sym_reflect(ampl_filter, f_kot_index, change_sign=False)
    general_filter = ampl_filter * 1
    mat_draw_functions(frequency_line, ampl_filter)
    return general_filter


def get_low_ampl(frequency_line, boundary_index):
    return np.concatenate((np.ones(boundary_index), np.zeros(len(frequency_line) - boundary_index)))


def get_high_ampl(frequency_line, boundary_index):
    return np.concatenate((np.zeros(boundary_index), np.ones(len(frequency_line) - boundary_index)))


def gcd(a, b):
    """Нахождение наибольшего общего делителя"""
    while b:
        a, b = b, a % b
    return a


def lcm(a, b):
    """Нахождение наименьшего общего кратного"""
    return a * b // gcd(a, b)


from fractions import Fraction


def discr_rel_number(signal, freq_coef):
    fraction = Fraction(freq_coef)
    numerator, denominator = fraction.numerator, fraction.denominator
    zeros_pad_array = add_zeros(signal, numerator - 1)
    spectrum = np.fft.fft(zeros_pad_array)
    spectrum = cut_center_in_array(spectrum, relative_cutted_part_size=numerator/(numerator + 1), pass_zeros=True)
    output_signal = take_every_step(np.fft.ifft(spectrum),  denominator)
    return output_signal * fraction.numerator

