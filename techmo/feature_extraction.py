#!/usr/bin/env python3

"""
feature_extraction.py: Extracts features based on on fft and wavelet transform
"""

__author__ = "Mariusz Źiółko, Michal Kucharski"
__email__ = "mariusz.ziolko@techmo.pl, michal.kucharski@techmo.pl"
__all__ = ['calculate_fft_wavelet', 'calculate_wavelet_fft']

import numpy as np
import pywt
import soundfile as sf


def apply_triangles(sig_freq_domain, triangle_no=20, sample_bucket_no=20):
    """
    Filtering method developed by prof. M. Źiółko
    It Applies :param `triangle_no` triangle filters to signal after fft
    :param sig_freq_domain: numpy vector representing signal after fft
    :param triangle_no: number of triangles, default 20
    :param sample_bucket_no: number of samples per each half of the triangle
    :return params: numpy array with length of :param `triangle_no`
    """
    sig_len = sig_freq_domain.shape[0]
    samples_per_half_triangle = sig_len // sample_bucket_no
    params = np.zeros((triangle_no,))
    for triangle_index in range(0, triangle_no - 2):
        for sample_index in range(0, samples_per_half_triangle + 1):
            sig_index = (triangle_index + 1) * samples_per_half_triangle + sample_index
            params[triangle_index + 1] += sample_index * sig_freq_domain[sig_index] ** 2 / samples_per_half_triangle
            params[triangle_index] += (1 - sample_index / samples_per_half_triangle) * sig_freq_domain[sig_index] ** 2

    # Logic for triangles on the edges
    for sample_index in range(0, samples_per_half_triangle + 1):
        params[0] += sample_index * sig_freq_domain[sample_index] ** 2 / samples_per_half_triangle
        sig_index = (triangle_no - 1) * samples_per_half_triangle + sample_index - 1
        params[triangle_no - 1] += (1 - sample_index / samples_per_half_triangle) * sig_freq_domain[sig_index] ** 2

    return np.log(params / (samples_per_half_triangle + 1))


def fft(normalized_signal):
    ft = np.fft.fft(normalized_signal)
    fft_len = ft.shape[0]
    mft = np.abs(ft[:fft_len // 2])
    return mft


def normalize_signal_by_energy(sig):
    return sig / np.sqrt(np.power(sig, 2).sum())


def enforce_sample_float_type(signal):
    numpy_arr_signal = np.array(signal)
    if numpy_arr_signal.dtype == np.int16:
        numpy_arr_signal = convert_integer_to_float(numpy_arr_signal)
    assert numpy_arr_signal.dtype == np.float
    return numpy_arr_signal


def convert_integer_to_float(signal):
    assert signal.dtype == np.int16
    int_to_float_divider = np.abs(np.iinfo(np.int16).min)
    return signal / int_to_float_divider


def calculate_fft_wavelet(wav_path, type='dmey', level=5):
    signal, fs = sf.read(wav_path)
    numpy_arr_signal = enforce_sample_float_type(signal)
    normalized_signal = normalize_signal_by_energy(numpy_arr_signal)
    mft = fft(normalized_signal)
    return np.stack([apply_triangles(coef) for coef in pywt.wavedec(mft, type, level=level)], axis=0)


def calculate_wavelet_fft(wav_path, type='dmey', level=5):
    signal, fs = sf.read(wav_path)
    numpy_arr_signal = enforce_sample_float_type(signal)
    normalized_signal = normalize_signal_by_energy(numpy_arr_signal)
    coefs = reversed(list(pywt.wavedec(normalized_signal, type, level=level)))
    return np.stack([apply_triangles(fft(coef)) for coef in coefs], axis=0)
