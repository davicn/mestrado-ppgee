import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, moment
import mne
from os.path import expanduser
import sys
sys.path.append('./')
from functions.support import aux,edfArray
from joblib import Parallel,delayed
import matplotlib.pyplot as plt

c1 = pd.DataFrame(
    data=np.load('data/emg_gnsz_train.npy'),
    columns=['path','inicio','fim','tipo','freq','tempo','montage'])

c2 = pd.DataFrame(
    data=np.load('data/emg_train_n_seizure.npy'),
    columns=['path','freq','tempo','montage'])

HOME = expanduser('~')+'/edf/'

# Carregando sinais
raws = Parallel(n_jobs=4)(
    delayed(edfArray)(
        HOME+c1.loc[i,'path'].replace('.tse','.edf'),'EMG') 
        for i in range(10))

# Calculando variância
v = Parallel(n_jobs=4)(
    delayed(aux)(i,np.var)
    for i in raws)

# Calculando variância
s = Parallel(n_jobs=4)(
    delayed(aux)(i,skew)
    for i in raws)

# Calculando variância
k = Parallel(n_jobs=4)(
    delayed(aux)(i,kurtosis)
    for i in raws)

fig = plt.figure()
plt.plot(v[0])
fig = plt.figure()
plt.plot(s[0])
fig = plt.figure()
plt.plot(k[0])
plt.show()

