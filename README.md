[![pypi](https://img.shields.io/pypi/v/techmo-wavelet.svg)](https://pypi.org/pypi/techmo-wavelet)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/techmo-wavelet.svg)](https://pypi.org/pypi/techmo-wavelet)
![example workflow](https://github.com/techmo-pl/vamp-wavelet-fft/actions/workflows/python-publish.yml/badge.svg)
## [Techmo Sp. z o.o.](http://techmo.pl) module for audio features extraction

### How to use
:warning: Add `!` character if you install the module in a jupyter notebook
```
pip install techmo-wavelet 

from techmo.feature_extraction import calculate_wavelet_fft
# install numpy first in case is not installed in your environment
import numpy as np 

# signal must be 1d array read from wav file, e.x by using Soundfile. Here we generate random signal
signal = np.random.uniform(-1.0, 1.0, 16000)

features = calculate_wavelet_fft(signal)
```


### The code implements an algorithm consisting of the following stages:
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
