import pandas as pd
import numpy as np
import sys

def drop_high_corr(percent):
	df = pd.read_csv("../Dataset/train/X_train.txt", delim_whitespace=True, header=None)
	header = pd.read_csv("../Dataset/features_fix.txt", delim_whitespace=True, header=None)
	df.columns = header[1]
	#Create correlation matrix
	matrix_corr = df.corr().abs()
	#Create correlaton matrix with only the upper triangle
	upper_tri = matrix_corr.where(np.triu(np.ones(matrix_corr.shape), k=1).astype(np.bool))
	#Features columns with correlation higher than given percent in decimal
	drop_features = [col for col in upper_tri.columns if any(upper_tri[col] > float(percent))]
	#Drop features with high correlation
	new_features = df.drop(drop_features, axis=1)
	print(f"Number of feature-pairs with correlation over {percent}: ", len(drop_features))
	print(f"Number of feature-pairs with correlation under {percent}: ", len(new_features.columns))
	return new_features

drop_high_corr(sys.argv[1])