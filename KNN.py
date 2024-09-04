import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from cvalidation import *
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold

y=pd.read_csv("/Users/resit/Desktop/UCI_HAR_dataset/train/y_train.txt",header=None)
X=pd.read_csv("/Users/resit/Desktop/UCI_HAR_dataset/train/X_train.txt",header=None,delim_whitespace=True)


#X_train,X_test,y_train,y_test=train_tesKt_split(X,y,test_size=0.25,random_state=0)


#feature scalling
def feature_scalling(X_train,y_train,X_test):
    sc_X=StandardScaler()
    X_train=sc_X.fit_transform(X_train)
    X_test=sc_X.transform(X_test)
    return X_train,X_test

#Creating KNN
def KNN(X_train, y_train, X_test):
    rs = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
    rs.fit(X_train,y_train)
    return rs.predict(X_test)
# Run crossvalidation
results, conf_matrix = run_kfold(X,y,KNN,feature_scalling,10,'macro')
print(conf_matrix)


with open('/Users/resit/Desktop/UCI_HAR_dataset/activity_labels.txt') as f:
        labels = [r.strip().split()[-1] for r in f]


labels = [label.replace('_', ' ').title() for label in labels]
np.set_printoptions(precision=2)
plot_confusion_matrix(conf_matrix, classes=labels)
plt.savefig('KNN-1.png',dpi=200,bbox_inches='tight')