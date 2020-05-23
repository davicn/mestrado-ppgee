import numpy as np
import pandas as pd 
import mne
import sys,os
sys.path.append('./')
from functions.support import edfArray,orgMontage

HOME = os.path.expanduser('~')+'/edf/'
PATH = os.getcwd()

c1 = pd.DataFrame(
    data=np.load(PATH+'/data/train_seizure.npy'),
    columns=['path','inicio','fim','tipo','freq','tempo','montage','tipo_montage'])

c1 = c1[(c1['freq']==256) & (c1['tipo']=='GNSZ')]
c1.index = np.arange(len(c1))

c2 = pd.DataFrame(
    data=np.load(PATH+'/data/train_n_seizure.npy'),
    columns=['path','freq','tempo','montage','tipo_montage'])


edf1 = HOME + c1.loc[0,'path'].replace('.tse','.edf')

m = edfArray(edf1)

print(orgMontage(m,c1.loc[0,'tipo_montage']))