import numpy as np
from os.path import expanduser
import sys
import pandas as pd
import mne
import matplotlib.pyplot as plt

sys.path.append('./')

from functions import support

""" train = np.load('data/train_seizure.npy')

path = '/media/davi/2526467b-9f07-4ce5-9425-6d2b458567b7/home/davi/edf/'


eeg = support.read_edf(path+train[0,0].replace('tse','edf')) 

signal = support.orgMontage(eeg,'01_tcp_ar')

info = mne.create_info(ch_names=list(signal.index),sfreq=256)
raw = mne.io.RawArray(signal.to_numpy(),info)

raw.plot(title='Data from arrays',show=True, block=True) 
 """


d = pd.DataFrame(
    data=np.load('data/train_seizure.npy'),
    columns=['path','start','end','event','freq','time','montage','type'])
print(d)
d.to_pickle('train_seizure.pkl')