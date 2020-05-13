# Script para  identificação de EDF's com EMG

#%% 
import numpy as np 
import pandas as pd 
import os 

path = os.getcwd().replace('scripts','')
path
# %%
w = pd.DataFrame(
            data=np.load(path+'data/train_seizure.npy'),
            columns=['path','inicio','fim','tipo','freq','tempo','montage'])

n = pd.DataFrame(
            data=np.load(path+'data/train_n_seizure.npy'),
            columns=['path','freq','tempo','montage'])


# %%

w_ = w['montage'].to_numpy()

index_w = []
for i in range(len(w_)):
    for ii in range(len(w_[i])):
        if 'EMG' in w_[i][ii]:
            index_w.append(i)

n_ = n['montage'].to_numpy()

index_n = []
for i in range(len(n_)):
    for ii in range(len(n_[i])):
        if 'EMG' in n_[i][ii]:
            index_n.append(i)



# %%

np.save('emg_train_seizure.npy',w.iloc[index_w,:].to_numpy())
np.save('emg_train_n_seizure.npy',n.iloc[index_n,:].to_numpy())
