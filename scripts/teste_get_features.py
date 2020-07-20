import numpy as np 
import pandas as pd 
import sys,os
from joblib import Parallel,delayed
import matplotlib.pyplot as plt 
import seaborn as sns

sys.path.append('./')

from functions import features,support

#%%
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

#%%

PATH = '/media/davi/2526467b-9f07-4ce5-9425-6d2b458567b7/home/davi/edf/'

w = pd.read_pickle('data/train_seizure4.pkl')
n = pd.read_pickle('data/train_n_seizure4.pkl')


w = w[(w['freq']==256) & (w['type']=='01_tcp_ar')]
w.index = np.arange(len(w))

n = n[(n['freq']==256) & (n['type']=='01_tcp_ar')]
n.index = np.arange(len(n))

print(n.shape)
print(w.shape)

#%%

r1 = Parallel(n_jobs=4)(delayed(job1)(i) for i in range(len(w)))

df1 = pd.concat(r1,axis=1)

t = get_end(df1.shape[1]//256,n)

r2 = Parallel(n_jobs=4)(delayed(job2)(i) for i in range(t[0]))

df2 = pd.concat(r2,axis=1)

df2 = df2.iloc[:,:df1.shape[1]]

#%%


vec1 = np.array([support.aux(df1.iloc[i].to_numpy(),np.var) for i in range(len(df1))])
vec2 = np.array([support.aux(df2.iloc[i].to_numpy(),np.var) for i in range(len(df2))])

# print(vec1.shape)
# print(vec2.shape)

vec1_ = support.med(vec1)
vec2_ = support.med(vec2)

y1 = np.repeat('Com crise',len(vec1_))
y2 = np.repeat('Sem crise',len(vec2_))

y = np.hstack((y1,y2))


print(vec1_.shape)
print(vec2_.shape)

# aux_df = pd.DataFrame(data=np.array([vec1,vec2]).T,columns=['Com crise','Sem crise'])

fig1 = plt.figure()
plt.plot(vec1_,'.',label='Com crise')
plt.plot(vec2_,'.',label='Sem crise')
plt.legend()

fig2 = plt.figure()

plt.boxplot([vec1_,vec2_],labels=['Com crise','Sem crise'],showfliers=False)

plt.show()
