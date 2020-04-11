import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 12]
plt.rcParams.update({'font.size' : 18})

# creating a simple signal with 2 frequencies
dt = 0.001
t = np.arange(0, 1, dt)
f = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*120*t)        # sum of 2 frequencies
f_clean = f                                             # clean 2 tone signal
f = f + 2.5*np.random.randn(len(t))                     # random noise addition

plt.plot(t, f, color='c', LineWidth='1.5', label='Noisy')
plt.plot(t, f_clean, color='k', Linewidth='2', label='Clean')
plt.xlim(t[0], t[-1])
plt.legend()
plt.show()

""" Denoise the noisy signal and try to obtain the clean one """

n = len(t)
fhat = np.fft.fft(f, n)                             # compute fft = (signal, data points)
                                                    # complex valued(magnitude&phase) fourier coefficients vector
PSD = fhat * np.conj(fhat) / n                      # Power spectrum (power per frequency) Density
freq =(1/(dt*n)) * np.arange(n)                     # create x-axis of frequencies
L = np.arange(1, np.floor(n/2), dtype='int')        # only plot the first half of

fig, axs = plt.subplots(2, 1)

plt.sca(axs[0])
plt.plot(t, f, color='c', LineWidth=1.5, label='Noisy')
plt.plot(t, f_clean, color='k', LineWidth=2, label='Clean')
plt.xlim(t[0], t[-1])
plt.legend()

plt.sca(axs[1])
plt.plot(freq[L], PSD[L], color='c', LineWidth =2, label='Noisy')
plt.xlim(freq[L[0]], freq[L[-1]])
plt.ylabel('Power Spectrum')
plt.xlabel('Hertz')
plt.legend()

plt.show()


""" Even though the signal is noisy; The Power Spectrum has 2 super-clean peaks(1st at 50Hz. and 2nd at 120Hz.)
    that means most of the power in noisy signal is in 50Hz and 120Hz 
    and then there is a bunch of noise in noise floor contributing to the jitter on the data(signal) 
    
    We can filter out; any fourier coefficient that is smaller than 100 -- just zero it out
    any fourier coefficient larger than 100 -- keep it 
    do Inverse Fourier Transform -- reconstruct denoised signal """
# Use PSD to filter out noise
indices = PSD > 100                 # frequences larger than 100 --- large vector with a lot of 0s two entries of 1
PSDclean = PSD * indices            # Zero out all others
fhat = indices * fhat               # Zero out small fourier coefficients in Y
ffilt = np.fft.ifft(fhat)

# Plots
fig, axs = plt.subplots(3, 1)

plt.sca(axs[0])
plt.plot(t, f, color='c', LineWidth=1.5, label='Noisy')
plt.plot(t, f_clean, color='k', LineWidth='2', label='clean')
plt.xlim(t[0], t[-1])
plt.legend()

plt.sca(axs[1])
plt.plot(t, ffilt, color='k', LineWidth=2, label='Filtered')
plt.xlim(t[0], t[-1])
plt.legend()

plt.sca(axs[2])
plt.plot(freq[L], PSD[L], color='c', LineWidth=2, label='Noisy')
plt.plot(freq[L], PSDclean[L], color='k', LineWidth=1.5, label='Filtered')
plt.xlim(freq[L[0]], freq[L[-1]])
plt.legend()

plt.show()