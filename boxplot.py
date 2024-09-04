from cvalidation import *
from log_reg import *
from SVM import *
import pandas
import matplotlib.pyplot as plt

def boxplot_comparison(score_type):

      with open('UCI_HAR_dataset/features_fix.txt') as f:
         header = [r.strip().split()[-1] for r in f]

      with open('UCI_HAR_dataset/activity_labels.txt') as f:
         labels = [r.strip().split()[-1] for r in f]

      data = pd.read_csv('UCI_HAR_dataset/train/X_train.txt',header=None,names=header,delim_whitespace=True)
      activities = pd.read_csv('UCI_HAR_dataset/train/y_train.txt',header=None,names=['activity_id'])

      models = [(MLP,lda_feature_selection),(MLP,PCA_feature_reduction),
      			(SVM,lda_feature_selection),(SVM,PCA_feature_reduction),
      			(logistic_regression,lda_feature_selection),(logistic_regression,PCA_feature_reduction),
      			(k_nearest_neighbors,lda_feature_selection),(k_nearest_neighbors,PCA_feature_reduction)]
      score = []
      names = ["LDA & MLP","PCA & MLP","LDA & SVM","PCA & SVM","LDA & log_reg","PCA & log_reg","LDA & KNN","PCA & KNN"]

      for classifier,feature_selection in models:
         results, conf_matrix = run_kfold(data,activities,classifier,feature_selection,10,'macro')
         score.append(results[score_type])

      #boxplot algorithm comparison
      fig = plt.figure(figsize=(10,6))
      fig.suptitle('Algorithm Comparison')
      ax = fig.add_subplot(111)
      plt.boxplot(score)
      ax.set_xticklabels(names,rotation=45)
      plt.show()
      plt.close('all')

boxplot_comparison("accuracy")

