[![pypi](https://img.shields.io/pypi/v/techmo.svg)](https://test.pypi.org/pypi/techmo)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/techmo.svg)](https://test.pypi.org/pypi/techmo)
![example workflow](https://github.com/mikuchar/techmo/actions/workflows/python-publish.yml/badge.svg)
## [Techmo Sp. z o.o.](http://techmp.pl) module for audio features extraction

### How to use
```
pip install techmo

from techmo.feature_extraction import calculate_wavelet_fft

features = calculate_wavelet_fft("path/to/audio_file")
```


### The code implements an algorithm consisting of the following stages:
1. Speech segment is processed by the Hann window,
2. The analyzed segment is normalized,
3. Speech segment is processed by the wavlet transform,
4. Each sub band is subjected to the Fast Fourier Transform,
5. Triangular filtration,
6. Logarithm of filter outputs.

A detailed presentation of the algorithm is presented in the paper
M.Ziołko, M.Kucharski, S.Pałka, B.Ziołko, K.Kaminski, I.Kowalska, A.Szpakowicz, J.Jamiołkowski, M.Chlabicz, M.Witkowski:
Fourier-Wavelet Voice Analysis Applied to Medical Screening Tests.
Proceedings of the INTERSPEECH 2021 (under review).
