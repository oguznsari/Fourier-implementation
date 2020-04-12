""" transfor data vectors into their Fourier coefficients
    we will approximate derivaties by taking advantage of FFT

    1st - take a function that we can analytically compute the exact derivative
    2nd - we are going to compare how accurate FFT is compared to analytic derivative
    and we also compare with simple finite difference derivative """

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 12]
plt.rcParams.update({'font.size': 10})

n = 64
L = 30                      # the length of our domain is gonna be 30
dx = L / n
x = np.arange(-L/2, L/2, dx, dtype ='complex_')
f = np.cos(x) * np.exp(-np.power(x, 2) / 25)                            # Function
    # cosine * decaying Gaussian function = Cosine in a Gaussian envelope -- use chain rule to get exact derivative
df = -(np.sin(x) * np.exp(-np.power(x, 2) / 25) + (2 / 25) * x * f)     # Derivative

""" Approximate derivative using Finite Difference -- it is really crude derivative to be honest (fk+1 - fk) / deltax
    this is not a good approximation of the derivative has error that scales like Delta X order Delta X error
    better job would be Central Difference or a Higher Order finite difference -- but this is just illustration """
dfFD = np.zeros(len(df), dtype='complex_')
for kappa in range(len(df) - 1):
    dfFD[kappa] = (f[kappa + 1] - f[kappa]) / dx

dfFD[-1] = dfFD[-2]

""" Derivative using FFT (spectral derivative)        = i * W * fhat = i * K * fhat
    use kappa when fourier transforming in space      K - spatial frequencies (wave numbers)
    use omega when fourier transforming in time       W - temporal frequencies
    fhat is vector of fourier coefficients            and Kappa is a vector of frequencies
    frequency weighted fourier coefficients * i --- inverse fourier this and recover the derivative of data
    at those discrete sample points """
fhat = np.fft.fft(f)
kappa = (2 * np.pi / L) * np.arange(-n/2, n/2)
kappa = np.fft.fftshift(kappa)                          # re-order fft frequencies
dfhat = kappa * fhat * (1j)
dfFFT = np.real(np.fft.ifft(dfhat))

""" Up shot here: All of these steps is very fast and very accurate:
    FFT = O(n*log(n))
    - create kappa vector
    - compute derivative = i * kappa * f 
    - inverse fourier transform to get the derivative back in spatial units 
    and  because these are complex numbers in fhat when multiply out and inverse fourier transform 
    we might have very very small machine precision imaginary parts that why we are just going to take 
    the real part of this inverse fourier transform -- just being careful """

# Plots
plt.plot(x, df.real, color='k', LineWidth=2, label='True Derivative')
plt.plot(x, dfFD.real, '--', color='y', LineWidth=1.5, label='Finite Difference')
plt.plot(x, dfFFT.real, '--', color='r', LineWidth=1.5, label='FFT Derivative')
plt.legend()
plt.show()

""" As you increase the data points(increase n) in signal which is decreasing deltax
    - Finite Difference Derivative does get more accurate but very slowly 
    whereas our Spectral Derivative (FFT Derivative) gets more accurate very rapidly """

""" Conclusion FFT is great for Spectral Derivatives of smooth functions whose derivatives are continuous
    or else we will get Gibbs phenomenon """