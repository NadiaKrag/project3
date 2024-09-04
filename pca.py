import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neural_network import MLPClassifier

def pca_feature_selection(X_train,y_train,X_test):
    
    y_train = y_train
    #Standardize values
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #Perform feature selection
    for i in range(1,561):
        pca = PCA(n_components = i)
        pca.fit(X_train)
        X_test_reduced = pca.transform(X_test)
        X_train_reduced = pca.transform(X_train)
        acc_explained_variance = sum(pca.explained_variance_ratio_)
        if acc_explained_variance > 0.945:
            print("number of components:", i , "-" , "with explained variance:", acc_explained_variance)
            break

    return X_train_reduced,X_test_reduced


    def multi_layer_perceptron(X_train, y_train, X_test):
    
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    mlp.fit(X_train, y_train) 
    return mlp.predict(X_test)