import os
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams.update({'font.size': 18})

dt = 0.001                                  # 1 kiloHertz sampling
t = np.arange(0, 2, dt)                     # 2 seconds of audio
f0 = 50
f1 = 250
t1 = 2
x = np.cos(2 * np.pi * t * (f0 + (f1-f0)*np.power(t, 2) / (3*t1**2)))
# audio signal: from low freq(50Hz) tone to high freq(250Hz) tone - quadratic chirp path

fs = 1 / dt
sd.play(2*x, fs)                    # playin the sound

plt.specgram(x, NFFT=128, Fs=1/dt, noverlap=120, cmap='jet_r')
# gabor window wideness, sampling rate of the original signal, how much overlap should each of Gabor windows have
plt.colorbar()
plt.xlabel('Time, s')
plt.ylabel('Frequency[Hz]')
# and the intensity(colormap) -- tells how much power is in each of those
plt.show()

"""
import librosa
y, sr = librosa.load('westworld-piano.mp3')     # y = signal, sr = sampling_rate

plt.specgram(y[0:1000000], NFFT=5000, Fs=sr, noverlap=400, cmap='jet_r')
plt.colorbar()
plt.show()

print(sr)
print(size(y))
sd.play(y[0:30000], sr)
"""

""" You could take the spectrogram and you could take the SVD of it;
    and you can see what is kind of the Eigen cords that are being played 
    Could use that for classification"""