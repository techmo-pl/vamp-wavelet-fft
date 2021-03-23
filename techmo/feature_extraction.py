#!/usr/bin/env python3

"""
The code implements an algorithm consisting of the following stages:
1.Speech segment is processed by the Hann window,
2.Analyzed segment is normalized,
3.Speech segment is processed by the wavelet transform,
4.Each subband is subjected to the Fast Fourier Transform,
5.Triangular filtration,
6.Logarithm of filter outputs.

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
import soundfile as sf
from scipy.signal import get_window


def normalize_signal(sig):
    sig = sig / np.max(np.abs(sig))
    return sig


def decomposition_number(sig):
    length = len(sig)
    if length >= 384:
        return 5
    if length >= 224:
        return 4
    if length >= 136:
        return 3
    if length >= 88:
        return 2
    if length >= 64:
        return 1
    raise ValueError("Segment is too short")


def wavelet_decomposition(signal, level):
    w_transform = pywt.wavedec(signal, 'dmey', level=level)
    return w_transform


def fourier_analysis(wavelet_decomp, decomp):
    mft = []
    for m in range(0, decomp):
        ft = np.fft.fft(wavelet_decomp[m])
        N = ft.shape[0]
        spe = np.abs(ft[:N // 2])
        mft.append(spe)
    return mft


def apply_filters(spectra, decomp):
    """
    Filtering method developed by Techmo Poland.
    It Applies
    :tuple 'spectra' presents FFT spectra computed for wavelet subbands,
    :scalar 'decomp' is equal to numer of wavelet decompositions, default 5,
    :vector 'size' presents numbers of samples for individual subbands,
    :vector 'features' includes 60 features of analysed speech segment,
    :vector 'amplitude' consits of amplitude spectra successively used for each subband,
    :vector 'numb_sampls[m]' shows the numbers of filter inputs[m] = "2*numb_sampls[m]+1".
    """
    no_subbands = decomp + 1
    size = np.zeros(no_subbands, dtype=int)
    numb_sampls = np.zeros(no_subbands, dtype=int)
    features = np.zeros(60, dtype=float)
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


def calculate_wavelet_fft(wav_path):
    signal, fs = sf.read(wav_path)
    sig_size = signal.shape[0]
    window = get_window("hann", sig_size, fftbins=True)
    windowed_signal = signal * window
    normalized_signal = normalize_signal(windowed_signal)
    decomp = decomposition_number(normalized_signal)
    w_transform = wavelet_decomposition(windowed_signal, decomp)
    spectra = fourier_analysis(w_transform, decomp + 1)
    filter_out = apply_filters(spectra, decomp)
    return np.log10(filter_out)
