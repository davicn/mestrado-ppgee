import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, auc, roc_curve


def metrics(cm):
    VP, FN, FP, VN = cm.ravel()
    names = ["ACU: ", "SENS: ", "ESP: "]
    print(names[0],
          np.round(100*(VP+VN)/cm.sum(),
                   decimals=3))
    print(names[1],
          np.round((100*VP/(VP+FN)), decimals=3))
    print(names[2],
          np.round((100*VN/(VN+FP)), decimals=3))


# def plot_cm(cm):
    # con_df = pd.DataFrame(data=cm, columns=['No', 'Yes'])
    # sns.heatmap(con_df, annot=True, cmap=plt.cm.Blues)
    # plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')


def conf_matrix(y, y_pred):
    m = confusion_matrix(y, y_pred)
    return m.astype('float')/m.sum(axis=1)[:, np.newaxis]


class Classifiers:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def knn(self):
        knn_ = KNeighborsClassifier(n_neighbors=5)
        y_pred = cross_val_predict(knn_, self.X, self.y, cv=10)
        # y_prob = cross_val_predict(knn_, self.X, self.y,  cv=10, method='predict_proba')
        cm = conf_matrix(self.y, y_pred)
        metrics(cm)
        # return roc_curve(self.y, y_prob[:, 1])
        #fpr_knn, tpr_knn, _ = C.knn()

    def svm(self):
        svc_ = SVC(kernel='rbf', probability=True)
        y_pred = cross_val_predict(svc_, self.X, self.y, cv=10)
        # y_prob = cross_val_predict(svc_, self.X, self.y, cv=10,method='predict_proba')
        cm = conf_matrix(self.y, y_pred)
        metrics(cm)
        # return roc_curve(self.y,y_prob[:,1])

    def svm_poly(self):
        svc_ = SVC(kernel='poly', probability=True)
        y_pred = cross_val_predict(svc_, self.X, self.y, cv=10)
        # y_prob = cross_val_predict(svc_, self.X, self.y, cv=10, method='predict_proba')
        cm = conf_matrix(self.y, y_pred)
        metrics(cm)
        # return roc_curve(y, y_prob[:, 1])

    def nb(self):
        nb_ = GaussianNB()
        y_pred = cross_val_predict(nb_, self.X, self.y, cv=10)
        # y_prob = cross_val_predict(nb_, self.X, self.y, cv=10, method='predict_proba')
        cm = conf_matrix(self.y, y_pred)
        metrics(cm)
        # return roc_curve(y, y_prob[:, 1])

    def lda(self):
        lda_ = LinearDiscriminantAnalysis()
        y_pred = cross_val_predict(lda_, self.X, self.y, cv=10)
        # y_prob = cross_val_predict(lda_, self.X, self.y, cv=10, method='predict_proba')
        cm = conf_matrix(self.y, y_pred)
        metrics(cm)
        # return roc_curve(y, y_prob[:, 1])

    def qda(self):
        qda_ = QuadraticDiscriminantAnalysis()
        y_pred = cross_val_predict(qda_, self.X, self.y, cv=10)
        # y_prob = cross_val_predict(qda_, self.X, self.y, cv=10, method='predict_proba')
        cm = conf_matrix(self.y, y_pred)
        metrics(cm)
        # return roc_curve(y, y_prob[:, 1])
