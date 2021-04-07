#!/usr/bin/env python3

"""
The code implements an algorithm consisting of the following stages:
1. Speech segment is processed by the Hann window,
2. The analyzed segment is normalized,
3. Speech segment is processed by the wavlet transform,
4. Each subband is subjected to the Fast Fourier Transform,
5. Triangular filtration,
6. Logarithm of filter outputs.
7. A feature vector of length 60 is returned

A detailed presentation of the algorithm is presented in the paper
M.Ziołko, M.Kucharski, S.Pałka, B.Ziołko, K.Kaminski, I.Kowalska, A.Szpakowicz, J.Jamiołkowski, M.Chlabicz, M.Witkowski:
Fourier-Wavelet Voice Analysis Applied to Medical Screening Tests.
Proceedings of the INTERSPEECH 2021 (under review).
"""

__author__ = "Mariusz Źiółko, Michal Kucharski"
__email__ = "mariusz.ziolko@techmo.pl, michal.kucharski@techmo.pl"
__all__ = ['calculate_wavelet_fft']

import pywt
import numpy as np
from scipy.signal import get_window


def _normalize_signal(sig):
    sig = sig / np.max(np.abs(sig))
    return sig


def _decomposition_number(sig):
    length = len(sig)
    if length >= 2048:
        return 5
    if length >= 1008:
        return 4
    if length >= 528:
        return 3
    if length >= 288:
        return 2
    if length >= 128:
        return 1
    raise ValueError("Segment is too short")


def _wavelet_decomposition(signal, level):
    w_transform = pywt.wavedec(signal, 'dmey', level=level)
    return w_transform


def _fourier_analysis(wavelet_decomp, decomp):
    mft = []
    for m in range(0, decomp):
        ft = np.fft.fft(wavelet_decomp[m])
        N = ft.shape[0]
        spe = np.abs(ft[:N // 2])
        mft.append(spe)
    return mft


def _apply_filters(spectra, decomp):
    """
    :param spectra: list of FFT spectra (ndarray) computed for wavelet subbands
    :param decomp: integer  is equal to numer of wavelet decompositions
    :return:
    """
    no_subbands = decomp + 1
    size = np.zeros(no_subbands, dtype=int)
    numb_sampls = np.zeros(no_subbands, dtype=int)
    features = np.zeros(60, dtype=float)

    """
    'size' presents numbers of samples for individual subbands,
    'features' includes 60 features of analysed speech segment,
    'amplitude' consits of amplitude spectra successively used for each subband,
    'numb_sampls[m]' shows the numbers of filter inputs[m] = "2*numb_sampls[m]+1".
    """

    for m in range(0, no_subbands):
        size[m] = spectra[m].shape[0]
        numb_sampls[m] = (size[m] * no_subbands - 2 * (decomp + 31)) / (decomp + 61)
        amplitude = spectra[m]
        for triang in range(0, 60 // no_subbands):
            feature_index = triang + m * 60 // no_subbands
            normalization_term = (numb_sampls[m] + 3)
            for sampl in range(0, numb_sampls[m]):
                features[feature_index] += (sampl + 1) * amplitude[triang * (numb_sampls[m] + 1) + sampl + 1] * 2 ** (
                        decomp - m - 1) / normalization_term
                features[feature_index] += (2 + numb_sampls[m] - sampl) / (1 + numb_sampls[m]) * amplitude[
                    triang * (numb_sampls[m] + 1) + 4 + sampl] * 2 ** (decomp - m - 1) / normalization_term
            features[feature_index] += amplitude[(numb_sampls[m] + 1) * (triang + 1)] * 2 ** (
                    decomp - m - 1) / normalization_term
    for triang in range(0, 60 // no_subbands):
        features[triang] = features[triang] / 2
    return features


def _enforce_sample_float_type(signal):
    numpy_arr_signal = np.array(signal)
    if numpy_arr_signal.dtype == np.int16:
        numpy_arr_signal = _convert_integer_to_float(numpy_arr_signal)
    assert numpy_arr_signal.dtype == float
    return numpy_arr_signal


def _convert_integer_to_float(signal):
    assert signal.dtype == np.int16
    int_to_float_divider = np.abs(np.iinfo(np.int16).min)
    return signal / int_to_float_divider


def calculate_wavelet_fft(signal):
    """
    :param signal:ndarray, 1d wave signal
    :return: ndarray, a feature vector of length 60
    """
    if len(signal.shape) != 1:
        raise ValueError('Signal must be a 1-dimensional array')
    signal = _enforce_sample_float_type(signal)
    sig_size = signal.shape[0]
    window = get_window("hann", sig_size, fftbins=True)
    windowed_signal = signal * window
    normalized_signal = _normalize_signal(windowed_signal)
    decomp = _decomposition_number(normalized_signal)
    w_transform = _wavelet_decomposition(windowed_signal, decomp)
    spectra = _fourier_analysis(w_transform, decomp + 1)
    filter_out = _apply_filters(spectra, decomp)
    return np.log10(filter_out)
