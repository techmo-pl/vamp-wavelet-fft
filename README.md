[![pypi](https://img.shields.io/pypi/v/techmo-wavelet.svg)](https://pypi.org/pypi/techmo-wavelet)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/techmo-wavelet.svg)](https://pypi.org/pypi/techmo-wavelet)
![example workflow](https://github.com/techmo-pl/vamp-wavelet-fft/actions/workflows/python-publish.yml/badge.svg)
## [Techmo Sp. z o.o.](http://techmo.pl) module for audio features extraction

### How to use
:warning: Add `!` character if you install the module in a jupyter notebook
```
pip install techmo-wavelet 

#import functions for feature extraction
from techmo.feature_extraction import calculate_wavelet_fft, calculate_fft_wavelet

# install numpy first in case is not installed in your environment
import numpy as np 

# signal must be 1d array read from wav file, e.x by using Soundfile. Here we generate random signal
signal = np.random.uniform(-1.0, 1.0, 16000)

# Here's an example of how to use `calculate_wavelet_fft` function
features = calculate_wavelet_fft(signal)

# Here's an example of how to use `calculate_fft_wavelet` function
features = calculate_fft_wavelet(signal)

```


### The code implements 2 functions to extract features:

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
7. The logarithms of filter outputs are computed to obtain
   the feature sub-vectors of length 60 for each segment.
8. Sub-vectors are concatenated to obtain a final feature matrix as numpy ndarray
   of shape int(N/2400), 60.


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
7. The computed modules are inputs of the triangular filtration,
8. The logarithms of filter outputs are computed to obtain
   the feature sub-vectors of length 60 for each segment.
9. Sub-vectors are concatenated to obtain a final feature matrix
   as numpy ndarray of shape int(N/4800), 60.