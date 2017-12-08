import numpy as np
# from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
np.set_printoptions(threshold=np.nan)
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

###########----------TRAINING DATA---------------###########
filename = "data/binarizedTraining.csv"

raw_data = pd.read_csv(filename,sep=',')

# print raw_data.ix[0]
X = np.asarray(raw_data)
print X.shape

# print "features\n"
# print X[0,:]

y = X[:,6]
X = X[:,7:]

X = np.asmatrix(X)

for i in range (0, y.shape[0]):
    if (y[i] == "Charged Off"):
        y[i] = 1
    else:
        y[i] = 0

X_standardize = X[:, 0:15]
X = X[:,15:]
X_standardize = preprocessing.scale(X_standardize)
X_standardize = np.concatenate((X_standardize, X), axis=1)


y = y.astype('int')

# X_standardize = X_standardize[0:120,:]
#
# pca = PCA(n_components='mle', svd_solver='full')
# print "fitting data to PCA"
#
# pca.fit(X_standardize)
# print "done fitting"
# print pca.components_.shape
# print "\n\n"
# print pca.explained_variance_ratio_
# print pca.explained_variance_ratio_.shape
#


# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# print X
# print "\n"
# pca = PCA(svd_solver='full')
# pca.fit(X)
# print(pca.explained_variance_ratio_)
#
# print(pca.singular_values_)
#

print "TREE CLASSIFIER"
clf = ExtraTreesClassifier(max_features =X_standardize.shape[1])
clf = clf.fit(X_standardize, y)
print "FEATURE IMPORTANCES"
print clf.feature_importances_.shape
print clf.feature_importances_
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X_standardize)
print "transformed X"
print X_new.shape
