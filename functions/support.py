import numpy as np
from numba import jit
import mne 


def aux(x, func):
    fs = 256
    m = np.zeros(x.shape[0]//fs)
    for i in range(len(m)):
        m[i] = func(x[i*fs:(i+1)*fs])
    return m 

def edfArray(path,label):
    raw = mne.io.read_raw_edf(path,preload=True).to_data_frame()
    sig = ''
    for i in raw.columns:
        if label in i:
            sig=i
    return raw.loc[:,sig].to_numpy()
