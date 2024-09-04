from cvalidation import *
from plt_config import *
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def PCA_feature_reduction(X_train,y_train,X_test):
    pca = PCA()
    X_train = pca.fit_transform(X_train,y_train)
    X_test = pca.transform(X_test)

    return X_train,X_test

def lda_feature_selection(X_train,y_train,X_test):
    lda = LDA()
    X_train = lda.fit_transform(X_train,y_train)
    X_test = lda.transform(X_test)

    return X_train,X_test

def SVM(X_train,y_train,X_test):
    svc = SVC(kernel="linear") #rbf, linear, poly
    svc.fit(X_train,y_train) #sample_weight=sample_weights
    return svc.predict(X_test)

def MLP(X_train,y_train,X_test):
    mlp = MLPClassifier()
    mlp.fit(X_train,y_train)
    return mlp.predict(X_test)

if __name__ == "__main__":
    ## Read in data
    with open('UCI_HAR_dataset/features_fix.txt') as f:
        header = [r.strip().split()[-1] for r in f]

    with open('UCI_HAR_dataset/activity_labels.txt') as f:
        labels = [r.strip().split()[-1] for r in f]

    data = pd.read_csv('UCI_HAR_dataset/train/X_train.txt',header=None,names=header,delim_whitespace=True)
    activities = pd.read_csv('UCI_HAR_dataset/train/y_train.txt',header=None,names=['activity_id'])
    new_data = data.drop(new_features,axis=1)

    ## Run crossvalidation
    results, conf_matrix = run_kfold(data,activities,MLP,PCA_feature_reduction,10,'macro')

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
    plt.savefig('PCA_SVM.png',dpi=200,bbox_inches='tight')