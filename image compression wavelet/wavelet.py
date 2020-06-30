""" instead of FFT using discrete Wavelet transform
    should be better at capturing multiple scales in these images (hair, fur and grass)"""

from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt

plt.rcParams['figure.figsize'] = [16, 16]
plt.rcParams.update({'font.size': 18})

A = imread('headshot.jpg')
B = np.mean(A, -1)                  # converts RGB to grayscale

# Wavelet decomposition (2 level - 2 dimensional)
n = 2                                                               # 2 level
w = 'db1'                                                           # daubechies wavelet family
coeffs = pywt.wavedec2(B, wavelet=w, level=n)

# Normalize each coefficient array
coeffs[0] /= np.abs(coeffs[0]).max()
for detail_level in range(n):
    coeffs[detail_level + 1] = [d/np.abs(d).max() for d in coeffs[detail_level + 1]]

arr, coeff_slices = pywt.coeffs_to_array(coeffs)

plt.imshow(arr, cmap='gray_r', vmin=-0.25, vmax=0.75)
plt.rcParams['figure.figsize'] = [16, 16]
plt.show()