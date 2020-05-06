""" Fourier Series --> Approximate periodic functions using an infinite sum of cosines and sine waves
    Gibbs Phenomenon happens while computing Fourier Series of a Discontinuous Function:
     Top-hat function discontinuous at edges of unit step """

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})

dx = 0.01                                                                       # 0.001
L = 2 * np.pi
x = np.arange(0, L+dx, dx)
n = len(x)
nquart = int(np.floor(n/4))

f = np.zeros_like(x)
f[nquart:3*nquart] = 1

A0 = np.sum(f * np.ones_like(x)) * dx * 2 / L
fFs = A0/2 * np.ones_like(f)

for k in range(1, 101):                                                         # 1001
    Ak = np.sum(f * np.cos(2 * np.pi * k * x / L)) * dx * 2 / L
    Bk = np.sum(f * np.sin(2 * np.pi * k * x / L)) * dx * 2 / L
    fFs = fFs + Ak * np.cos(2 * k * np.pi * x / L ) + Bk * np.sin(2 * k * np.pi * x / L)

plt.plot(x, f, color='k', LineWidth=2)
plt.plot(x, fFs, '-', color='r', LineWidth=1.5)
plt.show()


""" Fourier Series aprroximation with first 100 sines and cosines Gibbs Phenomenon at the corners where discontinuous
    Ringing behavior at the points of discontinuity 
    !! Reminder !! = If we added up all infinitely many sines and cosines of all frequencies we would have 
                     perfect approximation of discontinuous top-hat function 
                     
    !!! we have discontinuous function but sines and cosines are still continuous no sharp corners or jumps 
        these corners and sharps requires all fourier frequences to be able to approximate 

    Gibbs Phenomenon: happens while approximating discontinuous function using a finite truncated Fourier Series 
    
    If we use the Fourier Series to approximate the derivative of triangular hat function - which is a top hat function
     then the derivative is gonna have Gibbs Phenomenon """