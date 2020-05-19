import numpy as np
from os.path import expanduser
import sys
sys.path.append('./')
from functions.support import edfArray
from functions.features import FFT
import matplotlib.pyplot as plt 

home = expanduser('~')+'/edf/'

c = data = np.load('data/train_seizure.npy')

v = edfArray(home+c[0, 0].replace('.tse', '.edf'), 'EKG')


# xf, yf = FFT(v)
yf = FFT(v[:256])

# plt.plot(xf,yf)
plt.plot(yf)
plt.xlabel("Frequencia")
plt.ylabel("Amplitude")
plt.show()