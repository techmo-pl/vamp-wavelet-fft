#!/usr/bin/env python3

"""
The `calculate_wavelet_fft` function implements an algorithm consisting of the following stages:

1. If the number of samples N is greater than or equal to 4800,
  the signal is divided into int(N/2400) segments to compute finally 60
  features for each segment containing int(N/int(N/2400)) samples,
  i.e. the feature vector will have 60*int(N/2400) elements,
2. Segments are processed by the Hann window,
3. Segments are normalized separately,
4. Each segment is processed by the Wavelet Transform (WT),
5. Each WT subband is subjected to the Fast Fourier Transform (FFT),
6. FFT spectra are inputs of the triangular filtration to obtain
 the feature sub-vectors of length 60 for each segment,
7. The logarithms of filter outputs are computed.
8. Sub-vectors are concatenated to obtain a final feature matrix as numpy ndarray
  of shape int(N/2400), 60.

A detailed presentation of the algorithm is presented in the paper

M.Ziółko, M.Kucharski, S.Pałka, B.Ziółko, K.Kamiński, I.Kowalska,
A.Szpakowicz, J.Jamiołkowski, M.Chlabicz, M.Witkowski:
Fourier-Wavelet Voice Analysis Applied to Medical Screening Tests.
Proceedings of the INTERSPEECH 2021 (under review).

The `calculate_fft_wavelet` function implements an algorithm consisting of the following stages:

1. If the number of samples N is greater than or equal to 9600,
  the signal is divided into int(N/4800) segments to compute finally 60
  features for each segment containing int(N/int(N/4800)) samples,
  i.e. the feature vector will have 60*int(N/4800) elements,
2. Segments are processed by the Hann window,
3. Segments are normalized separately,
4. Speech segments are processed by the the Fast Fourier Transform,
5. The complex spectra are subjected to Wavelet Transform (WT),
6. Absolute values of WT are calculated,
7. The computed modules are inputs of the triangular filtration to obtain
  the feature sub-vectors of length 60 for each segment,
8. The logarithms of filter outputs are computed,
9. Sub-vectors are concatenated to obtain a final feature matrix
  as numpy ndarray of shape int(N/4800), 60.
"""

__author__ = "Mariusz Ziółko, Michał Kucharski"
__email__ = "mariusz.ziolko@techmo.pl, michal.kucharski@techmo.pl"
__all__ = ['calculate_wavelet_fft', 'calculate_fft_wavelet']

import pywt
import numpy as np
from scipy.signal import get_window


def _normalize_signal(sig):
    ekstrem = np.max(np.abs(sig))
    if ekstrem == 0:
        raise ValueError("All segment values are equal to 0")
    sig = sig / ekstrem
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


def _fast_fourier_transform(signal):
    ft = np.fft.fft(signal)
    N = ft.shape[0]
    return ft[:N // 2]


def _fourier_analysis_each_decomposition(wavelet_decomp, decomp):
    mft = []
    for m in range(0, decomp):
        spe = np.abs(_fast_fourier_transform(wavelet_decomp[m]))
        mft.append(spe)
    return mft


def _apply_filters(spectra, decomp):
    """
    :param spectra: list of spectra (ndarray) computed for the results of transform composition,
    :param decomp: integer is equal to number of wavelet decompositions,
    :return: vector for each subsegmnt of dimension 60.
    """
    no_subbands = decomp + 1
    size = np.zeros(no_subbands, dtype=int)
    numb_sampls = np.zeros(no_subbands, dtype=int)
    features = np.zeros(60, dtype=float)

    """
   'size' presents numbers of samples for individual subbands,
   'numb_sampls[m]' shows the numbers of filter inputs[m] = "2*numb_sampls[m]+1",
   'features' includes 60 features of analysed speech segment,
   'amplitude' consits of amplitude spectra successively used for each subband.
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


def _check_mono(signal):
    if len(signal.shape) != 1:
        raise ValueError('Signal must be a 1-dimensional array')


def calculate_wavelet_fft(signal):
    """
    :param signal:ndarray, 1d wave signal
    :return: ndarray, feature matrix with shape (1, 60) if the number N of samples is less than 4800,
             for the opposite case the feature matrix with shape  int(N/2400), 60
    """
    _check_mono(signal)
    signal = _enforce_sample_float_type(signal)
    sig_size = signal.shape[0]
    final_features = []
    no_seg = 1
    if sig_size >= 4800:
        no_seg = int(sig_size // 2400)
    segment = np.array_split(signal, no_seg)
    for m in range(0, no_seg):
        sig_size = segment[m].shape[0]
        window = get_window("hann", sig_size, fftbins=True)
        windowed_signal = segment[m] * window
        normalized_signal = _normalize_signal(windowed_signal)
        decomp = _decomposition_number(normalized_signal)
        w_transform = _wavelet_decomposition(normalized_signal, decomp)
        spectra = _fourier_analysis_each_decomposition(w_transform, decomp + 1)
        filter_out = _apply_filters(spectra, decomp)
        features = np.log10(filter_out)
        final_features.append(features)
    return np.stack(final_features)


def calculate_fft_wavelet(signal):
    """
    :param signal:ndarray, 1d wave signal
    :return: ndarray, feature matrix with shape (1, 60) if the number N of samples is less than 9600,
             for the opposite case the feature matrix with shape  int(N/4800), 60
    """
    _check_mono(signal)
    signal = _enforce_sample_float_type(signal)
    sig_size = signal.shape[0]
    final_features = []
    no_seg = 1
    if sig_size >= 9600:
        no_seg = int(sig_size // 4800)
    segment = np.array_split(signal, no_seg)
    for m in range(0, no_seg):
        sig_size = segment[m].shape[0]
        window = get_window("hann", sig_size, fftbins=True)
        windowed_signal = segment[m] * window
        normalized_signal = _normalize_signal(windowed_signal)
        spectrum = _fast_fourier_transform(normalized_signal)
        decomp = _decomposition_number(spectrum)
        w_transform = _wavelet_decomposition(spectrum, decomp)
        filter_in = [np.abs(comp) for comp in w_transform]
        filter_out = _apply_filters(filter_in, decomp)
        features = np.log10(filter_out)
        final_features.append(features)
    return np.stack(final_features)
