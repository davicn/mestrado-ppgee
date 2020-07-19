import numpy as np 
import pandas as pd 
import sys,os
from joblib import Parallel,delayed

sys.path.append('./')

from functions import features,support


def job1(index):
    i1 = int(w.loc[index,'start'])*w.loc[index,'freq']
    i2 = int(w.loc[index,'end'])*w.loc[index,'freq']

    eeg = support.read_edf(PATH+w.loc[index,'path'].replace('tse','edf')) 
    signal = support.orgMontage(eeg,w.loc[index,'type'])

    return signal.loc[:,i1:i2]

def job2(index):
    eeg = support.read_edf(PATH+n.loc[index,'path'].replace('tse','edf')) 
    signal = support.orgMontage(eeg,n.loc[index,'type'])
    return signal

def get_end(t,n):
    cont = 0
    for i in range(len(n)):
        if cont<=t:
            cont+=n.loc[i,'time']
        else:
            break
    return [i,cont]



PATH = '/media/davi/2526467b-9f07-4ce5-9425-6d2b458567b7/home/davi/edf/'

w = pd.read_pickle('data/train_seizure.pkl')
n = pd.read_pickle('data/train_n_seizure.pkl')

w = w.query("type=='01_tcp_ar'")
n = n[(n['freq']==256) & (n['type']=='01_tcp_ar')]
n.index = np.arange(len(n))



r1 = Parallel(n_jobs=4)(delayed(job1)(i) for i in range(50))

df1 = pd.concat(r1,axis=1)

t = get_end(df1.shape[1]//256,n)

r2 = Parallel(n_jobs=4)(delayed(job2)(i) for i in range(t[0]))

df2 = pd.concat(r2,axis=1)

df2 = df2.iloc[:,:df1.shape[1]]


