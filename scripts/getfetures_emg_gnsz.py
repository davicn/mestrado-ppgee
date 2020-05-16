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
import seaborn as sns


c1 = pd.DataFrame(
    data=np.load('data/emg_gnsz_train.npy'),
    columns=['path','inicio','fim','tipo','freq','tempo','montage'])

c2 = pd.DataFrame(
    data=np.load('data/emg_train_n_seizure.npy'),
    columns=['path','freq','tempo','montage'])


HOME = expanduser('~')+'/edf/'


# Carregando sinais
raws_w = Parallel(n_jobs=4)(
    delayed(edfArray)(
        HOME+c1.loc[i,'path'].replace('.tse','.edf'),'EMG') 
        for i in range(len(c1)))

raws_n = Parallel(n_jobs=4)(
    delayed(edfArray)(
        HOME+c2.loc[i,'path'].replace('.tse','.edf'),'EMG') 
        for i in range(len(c2)))

# Calculando variância
v_w = Parallel(n_jobs=4)(
    delayed(aux)(i,np.var)
    for i in raws_w)

v_n = Parallel(n_jobs=4)(
    delayed(aux)(i,np.var)
    for i in raws_n)

# Calculando Assimetria
s_w = Parallel(n_jobs=4)(
    delayed(aux)(i,skew)
    for i in raws_w)

s_n = Parallel(n_jobs=4)(
    delayed(aux)(i,skew)
    for i in raws_n)

# Calculando Curtose
k_w = Parallel(n_jobs=4)(
    delayed(aux)(i,kurtosis)
    for i in raws_w)
k_n = Parallel(n_jobs=4)(
    delayed(aux)(i,kurtosis)
    for i in raws_n)

# ______________________________________________________________________________
v_w_ = np.array([])
s_w_ = np.array([])
k_w_ = np.array([])

for i in range(len(c1)):
    start,end = c1.loc[i,['inicio','fim']].to_numpy().astype(int)
    v_w_ = np.append(v_w_,v_w[i][start:end])
    s_w_ = np.append(s_w_,s_w[i][start:end])
    k_w_ = np.append(k_w_,k_w[i][start:end])

col = ['var','skew','kur','class']

d1 = pd.DataFrame(
    data=np.array(np.array([v_w_,k_w_,s_w_,np.repeat('yes',len(v_w_))]).T),
    columns=col
)

# _______________________________________________________________________________
# Selecionando indices aleatórios
# para EMG sem crise
rand_index = lambda x1,x2:np.random.randint(x1,size=x2)

v_n_ = np.array([])
s_n_ = np.array([])
k_n_ = np.array([])

for i in range(len(c2)):
    v_n_ = np.append(v_n_,v_n[i])
    s_n_ = np.append(s_n_,s_n[i])
    k_n_ = np.append(k_n_,k_n[i])

d2 = pd.DataFrame(
    data=np.array(np.array([v_n_,k_n_,s_n_,np.repeat('no',len(v_n_))]).T),
    columns=col
)

r_ = rand_index(len(d2),len(d1))

d2 = d2.loc[r_,:]


# ________________________________________________________________________

d = pd.concat([d1,d2])
d.to_csv('testando.csv',index=False)

