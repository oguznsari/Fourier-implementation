from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt

plt.rcParams['figure.figsize'] = [16, 16]
plt.rcParams.update({'font.size': 18})

A = imread('headshot.jpg')
B = np.mean(A, -1)                      # convert RGB to grayscale

# Wavelet Compression
n = 4
w = 'db1'                               # daubechies wavelet
coeffs = pywt.wavedec2(B, wavelet=w, level=n)
coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs)
Csort = np.sort(np.abs(coeff_arr.reshape(-1)))

for keep in (0.1, 0.05, 0.01, 0.005):
    thresh = Csort[int(np.floor((1-keep)*len(Csort)))]
    ind = np.abs(coeff_arr) > thresh
    Cfilt = coeff_arr * ind                                         # threshold small indices

    coeffs_filt = pywt.array_to_coeffs(Cfilt, coeff_slices, output_format='wavedec2')

    # Plot reconstruction
    Arecon = pywt.waverec2(coeffs_filt, wavelet=w)
    plt.figure()
    plt.imshow(Arecon.astype('uint8'), cmap='gray_r')
    plt.axis('off')
    plt.rcParams['figure.figsize'] = [8, 8]
    plt.title('keep = ' + str(keep))
    plt.show()


""" Can code FFT2 vs Wavelet2 and compare to performance -- homework
    can compute the norm of the error to compare performance """