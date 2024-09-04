import pandas as pd
import os
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,confusion_matrix)

from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, GridSearchCV, RandomizedSearchCV, ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# from multiprocessing import Pool
import random
from pathos.multiprocessing import ProcessingPool as Pool
import dill
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from plt_config import *
import pandas as pd

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

class UCI_har_classification:
    def __init__(self, clf_args, feat_args, test_name, feat_chains=None, datadir='UCI_HAR_dataset', result_dir='results',average='weighted'):
        self._load_data(datadir)
        self.classifier_methods = {'KNN':KNeighborsClassifier,
                            'SVC':SVC, 
                            'MLP':MLPClassifier,
                            'LogReg': LogisticRegression,
                            'RF': RandomForestClassifier,
                            'GS': GridSearchCV
                            }
        self.feature_selection_methods = {'LDA':self.lda_feature_selection,
                                          'PCA':self.pca_feature_selection,    #### REPLACE WITH IDA
                                          'CORR':self.corr_feature_selection,  #### REPLACE WITH NADIA
                                          None:None
                                         }
        self.clf_args = clf_args
        self.feat_args = feat_args
        self.feat_chains = feat_chains
        self.kfold_args = None

        # self.results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        self.average = average

        # Directories and names for storing results
        self.test_name = test_name
        self.result_dir = result_dir
        self._create_result_dir()

    def _create_result_dir(self):
        i = 1
        dir_name = self.test_name
        while dir_name in os.listdir(self.result_dir):
            if not os.listdir(self.result_dir+'/'+dir_name):
                self.result_dir = '{}/{}'.format(self.result_dir,dir_name)
                return
            dir_name = self.test_name + '_{}'.format(i)
            i += 1
        self.result_dir = '{}/{}'.format(self.result_dir,dir_name)
        os.mkdir(self.result_dir)


    def _load_data(self,datadir):
        with open(datadir + '/features_fix.txt') as f:
            self.header = [r.strip().split()[-1] for r in f]

        with open(datadir + '/activity_labels.txt') as f:
            self.labels = [r.strip().split()[-1] for r in f]

        self.X_train = pd.read_csv(datadir + '/train/X_train.txt',
                                   header=None,
                                   names=self.header,
                                   delim_whitespace=True)

        self.y_train = pd.read_csv(datadir + '/train/y_train.txt',
                                   header=None,
                                   names=['activity_id'])

        self.subjects_train = pd.read_csv(datadir + '/train/subject_train.txt',
                                          header=None,
                                          names=['subject_id'])

    def lda_feature_selection(self,X_train,y_train,X_test,n_components=None):
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_train = lda.fit_transform(X_train,y_train)
        X_test = lda.transform(X_test)
        return X_train,X_test

    def corr_feature_selection(self,X_train,y_train,X_test,max_corr=0.95):
        #Create correlation matrix
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        matrix_corr = X_train.corr().abs()
        #Create correlaton matrix with only the upper triangle
        upper_tri = matrix_corr.where(np.triu(np.ones(matrix_corr.shape), k=1).astype(np.bool))
        #Features columns with correlation higher than given percent in decimal
        drop_features = [col for col in upper_tri.columns if any(upper_tri[col] > float(max_corr))]
        #Drop features with high correlation
        X_train = X_train.drop(drop_features, axis=1)
        X_test = X_test.drop(drop_features, axis=1)
        # print(f"Number of feature-pairs with correlation over {max_corr}: ", len(drop_features))
        # print(f"Number of feature-pairs with correlation under {max_corr}: ", len(X_train.columns))
        return X_train, X_test

    # def pca_feature_selection(X_train,y_train,X_test,min_var=0.95):
    def pca_feature_selection(self,X_train,y_train,X_test,min_var=0.95):
        #Perform feature selection
        pca = PCA()
        pca.fit(X_train)
        for i in range(1,len(pca.explained_variance_ratio_)):
            acc_explained_variance = sum(pca.explained_variance_ratio_[:i])
            if acc_explained_variance > min_var:
                # print("number of components:", i , "-" , "with explained variance:", acc_explained_variance)
                break

        pca = PCA(n_components=i)
        pca.fit(X_train)
        X_test_reduced = pca.transform(X_test)
        X_train_reduced = pca.transform(X_train)
        return X_train_reduced,X_test_reduced

    def get_uniq_train_subjects(self):
        return self.subjects_train.subject_id.unique()

    def get_pretty_labels(self):
        return [label.replace('_', ' ').title() for label in self.labels]

    # def print_results(self):
    #     for feat_name, clf_results in self.results.items():
    #         for clf_name,results in clf_results.items():
    #             print('{}+{} results:'.format(feat_name,clf_name))
    #             for measure, vals in results.items():
    #                 if measure == 'cm':
    #                     continue
    #                 print('{}: {:.4f}'.format(measure,np.mean(vals)))
    #             print()

    def print_ranking(self,metric='accuracy',to_file=None):
        d = {}
        for feat_name, feat_results in self.results.items():
            for clf_name,clf_results in feat_results.items():
                for param, results in clf_results.items():
                    d['{}+{}:{}'.format(feat_name,clf_name,set(param))] = np.mean(results[metric])
        import operator
        sorted_d = sorted(d.items(), key=operator.itemgetter(1),reverse=True)

        fname = self.result_dir + '/ranking_{}.txt'.format(metric)
        for i, (key, val) in enumerate(sorted_d):
            if to_file:
                print(i+1, val, key,file=open(fname,'a'))
            else:
                print(i+1, val, key)

    def get_person_split_iterator(self,n_splits,n_repeats):
        kf = RepeatedKFold(n_splits=n_splits,n_repeats=n_repeats)
        uniq_subjects = self.get_uniq_train_subjects()

        for train,test in kf.split(uniq_subjects):
            subjects_train = uniq_subjects[train]
            subjects_test = uniq_subjects[test]

            train_mask = self.subjects_train.subject_id.isin(uniq_subjects[train])
            test_mask = self.subjects_train.subject_id.isin(uniq_subjects[test])

            X_train = self.X_train[train_mask]
            y_train = self.y_train[train_mask].values.flatten()
            X_test = self.X_train[test_mask]
            y_test = self.y_train[test_mask].values.flatten()
            yield X_train, y_train, X_test, y_test

    def get_all_split_iterator(self,n_splits,n_repeats):
        skf = RepeatedStratifiedKFold(n_splits=n_splits,n_repeats=n_repeats)
        for train,test in skf.split(self.X_train, self.y_train):
            print(type(train),type(test))
            X_train = self.X_train.iloc[train]
            y_train = self.y_train.iloc[train].values.flatten()
            X_test = self.X_train.iloc[test]
            y_test = self.y_train.iloc[test].values.flatten()
            yield X_train, y_train, X_test, y_test

    def standard_scale(self, X_train, X_test):
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test

    def run_kfold(self,split='person',n_splits=21,n_repeats=1,scale=True,grid_search=False):
        count = 0

        if split=='person':
            split_iterator = self.get_person_split_iterator(n_splits,n_repeats)
        elif split=='all':
            split_iterator = self.get_all_split_iterator(n_splits,n_repeats)
        else:
            raise ValueError('{} is not a valid split method!'.format(split))

        for X_train_org, y_train, X_test_org, y_test in split_iterator:

            if scale:
                X_train_org, X_test_org = self.standard_scale(X_train_org,X_test_org)

            if count%n_splits == 0:
                print('{}. repeat:'.format(count//n_splits + 1),flush=True)
            print('\tFold {} of {}:'.format(count%n_splits + 1,n_splits),flush=True)
            for feat_name, feat_arg in self.feat_args.items():
                print('\t\tFeature selection: {}, Classifiers: '.format(feat_name),end='',flush=True)
                if self.feature_selection_methods[feat_name]:
                    X_train, X_test = self.feature_selection_methods[feat_name](X_train_org,y_train,X_test_org,**feat_arg)
                else:
                    X_train,X_test = X_train_org,X_test_org

                if grid_search:
                    y_preds,y_probas = self.grid_search(X_train, y_train, X_test, feat_name)
                else:
                    y_preds, y_probas = self.run_classifications(X_train, y_train, X_test, feat_name)

                self.add_results(y_test,y_preds,y_probas)


            if self.feat_chains:
                for chain in self.feat_chains:
                    chain_name = '+'.join(chain)
                    print('\t\tFeature selection: {}, Classifiers: '.format(chain_name),end='',flush=True)
                    X_train,X_test = X_train_org,X_test_org
                    for feat_name in chain:
                        X_train, X_test = self.feature_selection_methods[feat_name](X_train,y_train,X_test,**self.feat_args[feat_name])

                    if grid_search:
                        y_preds, y_probas = self.grid_search(X_train, y_train, X_test, chain_name)
                    else:
                        y_preds, y_probas = self.run_classifications(X_train, y_train, X_test, chain_name)

                    self.add_results(y_test,y_preds,y_probas)

            count += 1

        self.kfold_args = {'split':split,'n_splits':n_splits,'n_repeats':n_repeats,'average':self.average,'scale':scale}
        print('{}-fold cross-validation done with {} repeats!\n'.format(n_splits,n_repeats))

    def ROC(self, name_of_model):
        for feat_name, feat_results in self.results.items():
            for clf_name, clf_results in feat_results.items():
                for param, results in clf_results.items():
                    y_proba = np.concatenate(results['y_proba'],axis=0)
                    y_test = np.concatenate(results['y_test'],axis=0)
                    

        y_test_binarized = label_binarize(y_test,classes=[1,2,3,4,5,6])
        #y_preds_binarized = label_binarize(y_probas[feat_name][clf_name][param],classes=[1,2,3,4,5,6])
        n_classes = y_test_binarized.shape[1]


        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(0,6):
            #true_class = y_test_binarized[:,i]
            #class_proba = y_proba[:,i]
            fpr[i+1], tpr[i+1], _ = roc_curve(y_test_binarized[:,i],y_proba[:,i])
            roc_auc[i+1] = auc(fpr[i+1],tpr[i+1])


        #fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_preds_binarized.ravel())
        #roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        plt.figure()
        lw = 2
        #plt.plot(fpr[0],tpr[0],color="magenta",lw=lw,label="Class 1 (area = %0.2f)" % roc_auc[0])
        plt.plot(fpr[1],tpr[1],color="aquamarine",lw=lw,label="Walking (area = %0.3f)" % roc_auc[1])
        plt.plot(fpr[2],tpr[2],color="crimson",lw=lw,label="Walking Up (area = %0.3f)" % roc_auc[2])
        plt.plot(fpr[3],tpr[3],color="coral",lw=lw,label="Walking Down (area = %0.3f)" % roc_auc[3])
        plt.plot(fpr[4],tpr[4],color="lightblue",lw=lw,label="Sitting (area = %0.3f)" % roc_auc[4])
        plt.plot(fpr[5],tpr[5],color="plum",lw=lw,label="Standing (area = %0.3f)" % roc_auc[5])
        plt.plot(fpr[6],tpr[6],color="teal",lw=lw,label="Laying (area = %0.3f)" % roc_auc[6])
        plt.plot([0,1],[0,1],color="navy",lw=lw,linestyle="--")
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC for %s" %name_of_model)
        plt.legend(loc="lower right")
        plt.show()
        exit()
        
    

        #Compute macro-average ROC curve and ROC area

        #Aggregate all false positive

    def run_classifications(self, X_train, y_train, X_test, feat_name):
        y_preds = defaultdict(lambda:defaultdict(dict))
        y_preds_proba = defaultdict(lambda:defaultdict(dict))
        for clf_name, clf_arg in self.clf_args.items():
            print('{}... '.format(clf_name),end='',flush=True)
            y_pred, y_proba = self.run_classification(clf_name,X_train,y_train,X_test,clf_arg)
            y_preds[feat_name][clf_name][frozenset(clf_arg.items())] = y_pred
            y_preds_proba[feat_name][clf_name][frozenset(clf_arg.items())] = y_proba
        print()
        return y_preds, y_preds_proba

    def run_classification(self,clf_name, X_train, y_train, X_test, clf_arg):
        clf = self.classifier_methods[clf_name](**clf_arg)
        clf.fit(X_train,y_train)
        #print("PROBA")
        #print(clf.predict_proba(X_test))
        #print("PREDICT")
        #print(clf.predict(X_test))
        return clf.predict(X_test),clf.predict_proba(X_test)

    def grid_search(self, X_train, y_train, X_test, feat_name):
        y_preds = defaultdict(lambda:defaultdict(dict))
        y_preds_proba = defaultdict(lambda:defaultdict(dict))
        p = Pool()
        for clf_name, clf_arg in self.clf_args.items():
            print('{}... '.format(clf_name),end='',flush=True)
            grid_params = ParameterGrid(clf_arg)

            args = zip([clf_name]*len(grid_params),
                    [X_train]*len(grid_params),
                    [y_train]*len(grid_params),
                    [X_test]*len(grid_params),
                    grid_params
                    )
            results = p.map(self.run_classification,[clf_name]*len(grid_params),
                    [X_train]*len(grid_params),
                    [y_train]*len(grid_params),
                    [X_test]*len(grid_params),
                    grid_params
                    )

            for i, r in enumerate(results):
                y_preds[feat_name][clf_name][frozenset(grid_params[i].items())] = r[0]
                y_preds_proba[feat_name][clf_name][frozenset(grid_params[i].items())] = r[1]

        print()
        return y_preds, y_preds_proba

    def add_results(self,y_test, y_preds, y_probas):
        for feat_name, feat_preds in y_preds.items():
            for clf_name, clf_preds in feat_preds.items():
                for param, y_pred in clf_preds.items():
                    self.results[feat_name][clf_name][param]['accuracy'].append(accuracy_score(y_test,y_pred))
                    self.results[feat_name][clf_name][param]['precision'].append(precision_score(y_test,y_pred,average=self.average))
                    self.results[feat_name][clf_name][param]['recall'].append(recall_score(y_test,y_pred,average=self.average))
                    self.results[feat_name][clf_name][param]['f1_score'].append(f1_score(y_test,y_pred,average=self.average))

                    self.results[feat_name][clf_name][param]['cm'].append(confusion_matrix(y_test,y_pred))
                    
                    #self.results[feat_name][clf_name][param]['roc_curve'].append()
                    y_proba = y_probas[feat_name][clf_name][param]
                    self.results[feat_name][clf_name][param]['y_proba'].append(y_proba)
                    self.results[feat_name][clf_name][param]['y_test'].append(y_test)
                    #(y_proba)
                    #exit()

                    #self.ROC(y_test,y_proba)


                    #self.results[feat_name][clf_name][param]["fpr"].append(fpr)
                    #self.results[feat_name][clf_name][param]["tpr"].append(tpr)
                    #self.results[feat_name][clf_name][param]["auc"].append(roc_auc)


    def write_to_file(self):
        fname = self.result_dir + '/info.txt'
        with open(fname,'w') as f:
            f.write('Feature selection args:\n')
            for feat_name, feat_arg in self.feat_args.items():
                f.write('{}: {}\n'.format(feat_name,str(feat_arg)))
            f.write('\nClassifier args:\n')
            for clf_name, clf_arg in self.clf_args.items():
                f.write('{}: {}\n'.format(clf_name,str(clf_arg)))
            f.write('\nK-fold Cross-validation args:\n')
            f.write('{}\n'.format(str(self.kfold_args)))


    def save(self):
        fname = self.result_dir + '/uci_clf.pk'
        dill.dump(self,open(fname,'bw'))
        print('Saved instance in ' + fname)

    def save_results(self):
        fname = self.result_dir + '/results.pk'
        dill.dump(self,open(fname,'bw'))
        print('Saved results in ' + fname)

    @classmethod
    def load(self,f):
        return dill.load(open(f,'rb'))

    def plot_bar_chart(self):
        ## Flipping results since we want methods together in bar chart
        flipped_results = defaultdict(dict)
        for key, val in self.results.items():
            for subkey, subval in val.items():
                flipped_results[subkey][key] = subval 

        ## Get mean, std, and x_label for each method
        recall_means = []
        recall_std = []
        precision_means = []
        precision_std = []
        x_labels = []
        for clf_name, feat_results in flipped_results.items():
            for feat_name,results in feat_results.items():
                recall_means.append(np.mean(results['recall'])*100)
                recall_std.append(np.std(results['recall'])*100)
                precision_means.append(np.mean(results['precision'])*100)
                precision_std.append(np.std(results['precision'])*100)
                if feat_name:
                    x_labels.append(feat_name + '+' + clf_name)
                else:
                    x_labels.append(clf_name)


        fig,ax = newfig(1)
        ind = np.arange(len(recall_means))  # the x locations for the groups
        width = 0.2  # the width of the bars

        rects1 = ax.bar(ind - width/2, precision_means, width, yerr=precision_std,
                        color='dimgrey', label='Precision')
        rects2 = ax.bar(ind + width/2, recall_means, width, yerr=recall_std,
                        color='darkgrey', label='Recall')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        ax.set_xticks(ind)
        ax.set_xticklabels(x_labels,rotation=45,ha='right')
        ax.legend(bbox_to_anchor=(1,1))


        def autolabel(rects, xpos='center'):
            """
            Attach a text label above each bar in *rects*, displaying its height.

            *xpos* indicates which side to place the text w.r.t. the center of
            the bar. It can be one of the following {'center', 'right', 'left'}.
            """
            xpos = xpos.lower()  # normalize the case of the parameter
            ha = {'center': 'center', 'right': 'left', 'left': 'right'}
            offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.002*height,
                        '{:.1f}'.format(height), ha=ha[xpos], va='bottom', fontsize=9)


        autolabel(rects1, "left")
        autolabel(rects2, "right")

        fname = self.result_dir + '/performance_barchart.png'
        plt.savefig(fname,dpi=200,bbox_inches='tight')
        # print('Recognition performance bar chart saved as',fname,'\n')

    def plot_confusion_matrix(self):
        for feat_name, feat_results in self.results.items():
            for clf_name,clf_results in feat_results.items():
                for param, results in clf_results.items():
                    fname = self.result_dir + '/cm_{}_{}'.format(feat_name,clf_name)
                    for tup in param:
                        fname += '_{}-{}'.format(*tup)
                    self.plot_single_confusion_matrix(fname,sum(results['cm']))

    def plot_single_confusion_matrix(self, fname, cm):
        """
        This function plots the confusion matrix with precision and recall score 
        for each class.

        Adapted from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
        """
        np.set_printoptions(precision=2)
        classes = self.get_pretty_labels()

        mask = np.zeros(cm.shape,dtype=bool)
        np.fill_diagonal(mask,True)
        nondiag = np.ma.masked_array(cm,mask)
        diag = np.ma.masked_array(cm,~mask)

        fig,ax = newfig(1,0)

        pa = plt.imshow(diag,
                        interpolation='nearest',
                        cmap=plt.cm.Greens,
                        vmin=np.amin(cm),
                        vmax=np.amax(cm))
        pb = plt.imshow(nondiag,
                        interpolation='nearest',
                        cmap=plt.cm.Reds,
                        vmin=np.amin(cm),
                        vmax=np.amax(cm))

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

        fname += '.png'
        plt.savefig(fname,dpi=200,bbox_inches='tight')
        print('Confusion matrix plot saved as',fname,'\n')

    def plot_parameter_metrics(self,parameter):
        for feat_name, clf_results in self.results.items():
            for clf_name, args in self.clf_args.items():
                for param in ParameterGrid(args):
                    print(param, np.mean(self.results[feat_name][clf_name][frozenset(param.items())]['f1_score']))
                # fname = self.result_dir + '/cm_{}_{}'.format(feat_name,clf_name)
                # self.plot_single_confusion_matrix(fname,sum(results['cm']))




if __name__ == '__main__':
    # uci = UCI_har_classification.load('results/grid_search_SVC_2/uci_clf.pk')
    # uci.plot_parameter_metrics('n_neighbors')
    # uci.print_ranking('f1_score')
    # uci.plot_confusion_matrix()
    # # print(uci.results)
    # # uci.print_results()
    # exit()

    # Suppress warnings (you shouldn't do this)
    # import warnings
    # warnings.filterwarnings("ignore")

    # Create a classifier object
    uci = UCI_har_classification(clf_args={
                                            #'SVC':{'kernel':['linear', 'rbf','poly','sigmoid','precomputed'], 'C':[1,10]}
                                           #'SVC':{'kernel':['linear', 'rbf'], 'C':[0.0001,0.001,0.01,0.1,1,10,100]}
                                           # 'KNN':{'algorithm':['ball_tree','kd_tree'],'n_neighbors':range(90,100)}
                                           # 'KNN':{'n_neighbors':300}
                                           'LogReg':{}
                                           },
                                 feat_args={#'CORR':{'max_corr':0.90},
                                            #'LDA':{},
                                            # 'PCA':{'min_var':0.95},
                                            None:None,
                                            },
                                 #feat_chains = [('CORR','LDA')],
                                 test_name='grid_search_SVC',
            )

    # uci.grid_search()
    uci.run_kfold(n_splits=21,n_repeats=1,split='person',grid_search=False)
    uci.ROC('LogReg')
    # uci.print_results()
    uci.print_ranking('accuracy')
    uci.print_ranking('accuracy',to_file=True)
    # uci.print_ranking('recall')
    # uci.print_ranking('precision')
    uci.save()
    uci.save_results()
    uci.write_to_file()
