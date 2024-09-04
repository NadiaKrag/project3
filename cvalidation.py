import pandas as pd
import os
import itertools
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,confusion_matrix)
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from plt_config import *


def plot_confusion_matrix(cm, classes):
    """
    This function plots the confusion matrix with precision and recall score 
    for each class.

    Adapted from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    """
    print(cm)
    mask = np.zeros(cm.shape,dtype=bool)
    np.fill_diagonal(mask,True)
    nondiag = np.ma.masked_array(cm,mask)
    diag = np.ma.masked_array(cm,~mask)

    # fig,ax = plt.subplots()
    # plt.figure()
    fig,ax = newfig(1,0)

    pa = plt.imshow(diag,interpolation='nearest',cmap=plt.cm.Greens,vmin=np.amin(cm),vmax=np.amax(cm))
    pb = plt.imshow(nondiag,interpolation='nearest',cmap=plt.cm.Reds, vmin=np.amin(cm),vmax=np.amax(cm))
    # cbb = plt.colorbar(pb)#,shrink=0.25)
    # cba = plt.colorbar(pa)#,shrink=0.25)
    #
    # cba.set_label('Correct')
    # cbb.set_label('Incorrect')

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 verticalalignment='center',
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_xticklabels(ax.get_xticklabels(),ha='left')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    plt.ylabel('Groundtruth')
    plt.xlabel('Classification')

    text_color = '#696969'
    for i,j in enumerate(np.arange(1/len(classes)/2,1,1/len(classes))):
        precision = cm[i,i]/sum([row[i] for row in cm]) * 100
        recall = cm[i,i]/sum(cm[i]) * 100

        plt.text(j, -0.05, '{:.2f}'.format(precision), horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)#,color=text_color)
        plt.text(1.12, 1 - j,'{:.2f}'.format(recall), horizontalalignment='right',
            verticalalignment='center', transform=ax.transAxes)#,color=text_color)

    plt.text(1.075, 1.04,'Recall', horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes,color=text_color)
    plt.text(-0.025, -0.05,'Precision', horizontalalignment='right',
            verticalalignment='center', transform=ax.transAxes,color=text_color)

def run_kfold(X,y,classifier,feature_selection=None,n=10,average='micro'):
    skf = StratifiedKFold(n_splits=n,shuffle=True)

    results = defaultdict(list)
    count = 1
    for train, test in skf.split(X, y):
        print('Running fold',count,'of',n)
        X_train = X.iloc[train]
        X_test = X.iloc[test]
        y_train = y.iloc[train].values.flatten()
        y_test = y.iloc[test].values.flatten()

        if feature_selection:
            X_train, X_test = feature_selection(X_train,y_train,X_test)

        y_pred = classifier(X_train,y_train,X_test)

        results['accuracy'].append(accuracy_score(y_test,y_pred))
        results['precision'].append(precision_score(y_test,y_pred,average=average))
        results['recall'].append(recall_score(y_test,y_pred,average=average))
        results['f1_score'].append(f1_score(y_test,y_pred,average=average))

        try:
            conf_matrix += confusion_matrix(y_test,y_pred)
        except NameError:
            conf_matrix = confusion_matrix(y_test,y_pred)
        count += 1
    return results, conf_matrix

def lda_feature_selection(X_train,y_train,X_test):
    lda = LDA()
    X_train = lda.fit_transform(X_train,y_train)
    X_test = lda.transform(X_test)

    return X_train,X_test

def logistic_regression(X_train, y_train, X_test):
    lr = LogisticRegression()
    lr.fit(X_train,y_train)
    return lr.predict(X_test)

def k_nearest_neighbors(X_train, y_train, X_test):
    knn = KNeighborsClassifier(1)
    knn.fit(X_train,y_train)
    return knn.predict(X_test)



if __name__ == '__main__':
    ## Read in data
    with open('UCI_HAR_dataset/features_fix.txt') as f:
        header = [r.strip().split()[-1] for r in f]

    with open('UCI_HAR_dataset/activity_labels.txt') as f:
        labels = [r.strip().split()[-1] for r in f]

    data = pd.read_csv('UCI_HAR_dataset/train/X_train.txt',header=None,names=header,delim_whitespace=True)
    activities = pd.read_csv('UCI_HAR_dataset/train/y_train.txt',header=None,names=['activity_id'])

    ## Run crossvalidation
    results, conf_matrix = run_kfold(data,activities,k_nearest_neighbors,lda_feature_selection,10,'macro')

    ## Print result metrics
    for measure, vals in results.items():
        print('{}: {:.4f}'.format(measure,np.mean(vals)))

    # conf_matrix = np.array([[408,  1,  0,  0,  0,  0],
    #                         [ 31,303, 24,  0,  0,  0],
    #                         [ 15,  0,314,  0,  0,  0],
    #                         [  0,  2,  0,409, 18,  0],
    #                         [  0,  4,  0, 30,424,  0],
    #                         [  0,  0,  3,  1,  0,465]])

    ## Plot and save confusion matrix
    labels = [label.replace('_', ' ').title() for label in labels]
    np.set_printoptions(precision=2)
    plot_confusion_matrix(conf_matrix, classes=labels)
    plt.savefig('lda_knn.png',dpi=200,bbox_inches='tight')
