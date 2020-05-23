import numpy as np
from os.path import expanduser
import sys
sys.path.append('./')
# from functions.support import edfArray
# from functions.features import FFT
from functions.classifiers import Classifiers 

import matplotlib.pyplot as plt 

# home = expanduser('~')+'/edf/'
# 
# c = data = np.load('data/train_seizure.npy')
# 
# v = edfArray(home+c[0, 0].replace('.tse', '.edf'), 'EKG')
# 
# 
# xf, yf = FFT(v)
# yf = FFT(v[:256])
# 
# plt.plot(xf,yf)
# plt.plot(yf)
# plt.xlabel("Frequencia")
# plt.ylabel("Amplitude")
# plt.show()

from sklearn import datasets

iris = datasets.load_breast_cancer()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target

C = Classifiers(X,y)
C.knn()
C.lda()
C.nb()
C.qda()
