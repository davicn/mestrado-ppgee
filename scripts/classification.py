import numpy as np 
import pandas as pd 
import sys
sys.path.append('./')
from functions.classifiers import Classifiers 
from sklearn.preprocessing import LabelEncoder

d = pd.read_csv('notebooks/testando.csv')

X = d.iloc[:,:-1]

le = LabelEncoder()
le.fit(d['class'].to_numpy())
y = le.transform(d['class'].to_numpy())

C = Classifiers(X,y)
C.knn()
C.lda()
