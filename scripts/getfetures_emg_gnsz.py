import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, moment
import mne
import sys
sys.path.append('./')
from functions import support


c1 = pd.DataFrame(
    data=np.load('data/emg_gnsz_train.npy'),
    columns=['path','inicio','fim','tipo','freq','tempo','montage'])

c2 = pd.DataFrame(
    data=np.load('data/emg_train_n_seizure.npy'),
    columns=['path','freq','tempo','montage'])

print(c2.head())

