#code here
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
#np.set_printoptions(threshold=np.nan)
from sklearn import preprocessing
import matplotlib.pyplot as plt

from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler




filename = "data/binarizedTraining.csv"

raw_data = pd.read_csv(filename,sep=',')

X = np.asarray(raw_data)

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


#######----------TEST DATA--------------##########
filename = "data/binarizedTest.csv"

raw_data = pd.read_csv(filename,sep=',')

Xtest = np.asarray(raw_data)

ytest = Xtest[:,6]
Xtest = Xtest[:,7:]

Xtest = np.asmatrix(Xtest)

for i in range (0, ytest.shape[0]):
    if (ytest[i] == "Charged Off"):
        ytest[i] = 1
    else:
        ytest[i] = 0

#_________________________________________________#####        
        
Xtest_standardize = Xtest[:, 0:15]
Xtest = Xtest[:,15:]
Xtest_standardize = preprocessing.scale(Xtest_standardize)
Xtest_standardize = np.concatenate((Xtest_standardize, Xtest), axis=1)

print Xtest_standardize.shape
y = y.astype('int')

n_components = [15, 20, 25]

Cs = [0.1,0.01,0.001,0.0001,1]

logistic = linear_model.LogisticRegression()

pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

pca.fit(X_standardize)

estimator = GridSearchCV(pipe, 
                         dict(pca__n_components=n_components,
                              logistic__C=Cs), refit=True)
estimator.fit(np.asarray(X_standardize), y)

#weight vector of contributing factors of variance?
#under/oversampling
#area under the curve
#roc curve

print estimator.best_params_

pred = estimator.predict(Xtest_standardize)

standard_probs = estimator.predict_proba(Xtest_standardize)

over_sampling = True
under_sampling = False


if over_sampling:
    ros = RandomOverSampler(random_state=0)
    X_standardize, y = ros.fit_sample(training_data.reshape(-1,1), X_standardize.reshape(-1,1))
    X_standardize = [X_standardize.item(i) for i in range(len(X_standardize))]
elif under_sampling:
    rus = RandomUnderSampler(random_state=0)
    X_standardize, y = rus.fit_sample(training_data.reshape(-1,1), y.reshape(-1,1))
    X_standardize = [training_data.item(i) for i in range(len(training_data))]


training_probabilities = estimator.predict_proba(X_standardize)
test_probabilities = estimator.predict_proba(Xtest_Standardize)
print("Training AUC: {}".format(roc_auc_score(y, training_probabilities[:,1])))
print("Test AUC: {}".format(roc_auc_score(ytest, test_probabilities[:,1])))


#f = open('standard_logreg.txt', 'w+')
#for prob in standard_probs:
#    f.write(str(prob[0])+","+str(prob[1])+"\n")

f = open('OverSampled_train_logreg.txt', 'w+')
for prob in training_probabilities:
    f.write(str(prob[0])+","+str(prob[1])+"\n")

f = open('OverSampled_test_logreg.txt', 'w+')
for prob in test_probabilities:
    f.write(str(prob[0])+","+str(prob[1])+"\n")
    

    


    
    
    
#plt.figure(1, figsize=(4, 3))
#plt.clf()
#plt.axes([.2, .2, .7, .7])
#plt.plot(pca.explained_variance_, linewidth=2)
#plt.axis('tight')
#plt.xlabel('n_components')
#plt.ylabel('explained_variance_')
#
#plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
#            linestyle=':', label='n_components chosen')
#plt.legend(prop=dict(size=12))
#plt.save()