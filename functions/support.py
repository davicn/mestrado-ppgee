import numpy as np
from numba import jit


def aux(x, func):
    fs = 256
    m = np.zeros(x.shape[0]//fs)
    for i in range(len(m)):
        m[i] = func(x[i*fs:(i+1)*fs])
    return m 
