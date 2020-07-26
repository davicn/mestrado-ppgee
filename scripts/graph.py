import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from joblib import Parallel, delayed






d = pd.read_csv('/home/davi/Github/mestrado-ppgee/lista.txt',header=None)
path = '/home/davi/Github/mestrado-ppgee/'

l = [pd.read_pickle(path+d.iloc[i,0]) for i in range(len(d))]
data = pd.concat(l)


# def plotar(i):
    # print(i)
# plt.plot(data.query("Class == 'Com crise'").iloc[:,0].to_numpy(),'.')
# plt.plot(data.query("Class == 'Sem crise'").iloc[:,0].to_numpy(),'.')

aux = data.sort_values(by='Variance')

sns.boxplot(data=aux.iloc[:100],x='Class',y='Variance',hue='Class')
plt.plot()
# plt.show()

# Parallel(n_jobs=2)(delayed(plotar)(i) for i in range(1))

# plt.plot(data.query("Class == 'Com crise'").iloc[:,0])