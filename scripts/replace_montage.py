import numpy as np
import pandas as pd 
import mne
import sys,os
sys.path.append('./')
from functions.support import edfArray

HOME = os.path.expanduser('~')+'/edf/'
PATH = os.getcwd()

c1 = pd.DataFrame(
    np.load(PATH+'/data/train_seizure.npy'),
    columns=['path','inicio','fim','tipo','freq','tempo','montage'])

c1 = c1[(c1['freq']==256) & (c1['tipo']=='GNSZ')]
c1.index = np.arange(len(c1))

m, num = np.unique(c1['montage'],return_counts=True)

print(num)