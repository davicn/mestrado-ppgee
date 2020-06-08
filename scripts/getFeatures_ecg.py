from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt
# from functions.features import FFT, energy
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, moment
import mne
from os.path import expanduser
import sys
sys.path.append('./')
from functions.support import aux, edfArray


c1 = pd.DataFrame(
    data=np.load('data/ecg_train_seizure.npy'),
    columns=['path', 'inicio', 'fim', 'tipo', 'freq', 'tempo', 'montage'])

c2 = pd.DataFrame(
    data=np.load('data/ecg_train_n_seizure.npy'),
    columns=['path', 'freq', 'tempo', 'montage'])

c1 = c1[c1['freq'] == 256]
c1.index = np.arange(len(c1))

c2 = c2[c2['freq'] == 256]
c2.index = np.arange(len(c2))

# c1 = c1.iloc[:10]
# c2 = c2.iloc[:10]

HOME = expanduser('~')+'/edf/'


# # Carregando sinais
raws_w = Parallel(n_jobs=4)(
    delayed(edfArray)(
        HOME+c1.loc[i, 'path'].replace('.tse', '.edf'), 'EKG')
    for i in range(10))

raws_n = Parallel(n_jobs=4)(
    delayed(edfArray)(
        HOME+c2.loc[i, 'path'].replace('.tse', '.edf'), 'EKG')
    for i in range(10))

# # Calculando FFT
# fft_w = Parallel(n_jobs=4)(
#     delayed(aux)(i, FFT)
#     for i in raws_w)

# fft_n = Parallel(n_jobs=4)(
#     delayed(aux)(i, FFT)
#     for i in raws_n)

# # Calculando variância
v_w = Parallel(n_jobs=4)(delayed(aux)(i,kurtosis) for i in raws_w)

v_n = Parallel(n_jobs=4)(delayed(aux)(i,kurtosis) for i in raws_n)

print([len(i) for i in v_n])
# fig1 = plt.figure()
# plt.plot(v_n,'o')
# plt.plot(v_w,'o')

# fig2 = plt.figure()
# plt.boxplot([v_w,v_n])

# plt.show()
# # Calculando Assimetria
# s_w = Parallel(n_jobs=4)(
#     delayed(aux)(i, skew)
#     for i in raws_w)

# s_n = Parallel(n_jobs=4)(
#     delayed(aux)(i, skew)
#     for i in raws_n)

# # Calculando Curtose
# k_w = Parallel(n_jobs=4)(
#     delayed(aux)(i, kurtosis)
#     for i in raws_w)
# k_n = Parallel(n_jobs=4)(
#     delayed(aux)(i, kurtosis)
#     for i in raws_n)

# # ______________________________________________________________________________
# v_w_ = np.array([])
# s_w_ = np.array([])
# k_w_ = np.array([])
# fft_w_ = np.array([])


# for i in range(len(c1)):
#     start, end = c1.loc[i, ['inicio', 'fim']].to_numpy().astype(int)
#     v_w_ = np.append(v_w_, v_w[i][start:end])
#     s_w_ = np.append(s_w_, s_w[i][start:end])
#     k_w_ = np.append(k_w_, k_w[i][start:end])
#     fft_w_ = np.append(fft_w_, fft_w[i][start:end])

# col = ['var', 'skew', 'kur', 'fft', 'class']

# d1 = pd.DataFrame(
#     data=np.array(
#         np.array([v_w_, k_w_, s_w_, fft_w_, np.repeat('yes', len(v_w_))]).T),
#     columns=col
# )

# # _______________________________________________________________________________
# # Selecionando indices aleatórios
# # para EMG sem crise


# def rand_index(x1, x2): return np.random.randint(x1, size=x2)


# v_n_ = np.array([])
# s_n_ = np.array([])
# k_n_ = np.array([])
# fft_n_ = np.array([])


# for i in range(len(c2)):
#     v_n_ = np.append(v_n_, v_n[i])
#     s_n_ = np.append(s_n_, s_n[i])
#     k_n_ = np.append(k_n_, k_n[i])
#     fft_n_ = np.append(fft_n_, fft_n[i])

# d2 = pd.DataFrame(
#     data=np.array(
#         np.array([v_n_, k_n_, s_n_, fft_n_, np.repeat('no', len(v_n_))]).T),
#     columns=col
# )

# r_ = rand_index(len(d2), len(d1))

# d2 = d2.loc[r_, :]


# # ________________________________________________________________________

# d = pd.concat([d1, d2])
# d.to_csv('testando.csv', index=False)
# #
# #
