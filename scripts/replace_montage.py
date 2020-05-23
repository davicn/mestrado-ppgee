import numpy as np
import pandas as pd 
import mne
import sys,os
sys.path.append('./')
from functions.support import edfArray

HOME = os.path.expanduser('~')+'/edf/'
PATH = os.getcwd()

c1 = pd.DataFrame(
    data=np.load(PATH+'/data/train_seizure.npy'),
    columns=['path','inicio','fim','tipo','freq','tempo','montage'])

c1 = c1[(c1['freq']==256) & (c1['tipo']=='GNSZ')]
c1.index = np.arange(len(c1))

c2 = pd.DataFrame(
    data=np.load(PATH+'/data/train_n_seizure.npy'),
    columns=['path','freq','tempo','montage'])

tm1 = c1.loc[:,'path'].apply(lambda x: x.replace('train/','')[:9])
tm2 = c2.loc[:,'path'].apply(lambda x: x.replace('train/','')[:9])

# unique, cont = np.unique(tm,return_counts=True)
# print(unique)
# print(cont)
# 
# print(tm.shape)
# print(c1.shape)

c2['tipo_montage'] = tm2
c1['tipo_montage'] = tm1

np.save("train_seizure.npy",c1.to_numpy())
np.save("train_n_seizure.npy",c2.to_numpy())
