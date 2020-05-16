import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, moment
import mne
from os.path import expanduser
import sys
sys.path.append('./')
from functions.support import aux,edfArray


c1 = pd.DataFrame(
    data=np.load('data/emg_gnsz_train.npy'),
    columns=['path','inicio','fim','tipo','freq','tempo','montage'])

c2 = pd.DataFrame(
    data=np.load('data/emg_train_n_seizure.npy'),
    columns=['path','freq','tempo','montage'])

HOME = expanduser('~')+'/edf/'

for i in range(100):
    raw = edfArray(HOME+c1.loc[i,'path'].replace('.tse','.edf'),'EMG')
    v = aux(raw,np.var)
    s = aux(raw,skew)
    k = aux(raw,kurtosis) 
    



