import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift

FS = 256

def FFT(x):
    # return np.abs(np.array([fftshift(fftfreq(FS)),fftshift(fft(x))]))
    return fftshift(fft(x))
