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
from sklearn.model_selection import train_test_split



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
ytest = ytest.astype('int')

n_components = [20]

Cs = [0.001,0.0001]

logistic = linear_model.LogisticRegression()

pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

pca.fit(X_standardize)

estimator = GridSearchCV(pipe, 
                         dict(pca__n_components=n_components,
                              logistic__C=Cs), refit=True)
estimator.fit(np.asarray(X_standardize), y.ravel())

#weight vector of contributing factors of variance?
#under/oversampling
#area under the curve
#roc curve

print estimator.best_params_

include_test = False
test_percentage = 0.20
if (include_test):
    training_data, test_data, training_target, test_target = train_test_split(X_standardize, y, test_size=test_percentage,random_state=0)
else:
    training_data = X_standardize
    training_target = y
    test_data = Xtest_standardize
    test_target = ytest

over_sampling = True
under_sampling = False

if over_sampling and include_test:
    ros = RandomOverSampler(random_state=0)
    training_data, training_target = ros.fit_sample(training_data, training_target.reshape(-1,1))
    test_data, test_target = ros.fit_sample(test_data, test_target.reshape(-1,1))
#    X_standardize = [X_standardize.item(i) for i in range(len(X_standardize))]
    
elif under_sampling and include test:
    rus = RandomUnderSampler(random_state=0)
    training_data, training_target = rus.fit_sample(training_data, training_target.reshape(-1,1)) 
    test_data, test_target = rus.fit_sample(test_data, test_target.reshape(-1,1))

#    X_standardize = [training_data.item(i) for i in range(len(training_data))]

if over_sampling:
    print "over sampling:"
else:
    print "under sampling:"
pred = estimator.predict(training_data)
training_accuracy = accuracy_score(training_target, pred)
test_pred = estimator.predict(test_data)
test_accuracy = accuracy_score(test_target, test_pred)
print "training accuracy: ", training_accuracy
print "test accuracy: ", test_accuracy

#if over_sampling:
#    ros = RandomOverSampler(random_state=0)
#    X_standardize, y = ros.fit_sample(training_data, y.reshape(-1,1))
##    X_standardize = [X_standardize.item(i) for i in range(len(X_standardize))]
#elif under_sampling:
#    rus = RandomUnderSampler(random_state=0)
#    X_standardize, y = rus.fit_sample(X_standardize.reshape(-1,1), y.reshape(-1,1))
##    X_standardize = [training_data.item(i) for i in range(len(training_data))]


training_probabilities = estimator.predict_proba(np.array(training_data))
test_probabilities = estimator.predict_proba(np.array(test_data))


#print "Training AUC: " 
#print roc_auc_score(training_target, training_probabilities[:,1])
#print "--------------------------------------------"
#print("Test AUC: {}".format(roc_auc_score(test_target, test_probabilities[:,1])))


#f = open('standard_logreg.txt', 'w+')
#for prob in standard_probs:
#    f.write(str(prob[0])+","+str(prob[1])+"\n")

if over_sampling:
    f = open('OverSampled_train_logreg.txt', 'w+')
    for prob in training_probabilities:
        f.write(str(prob[0])+","+str(prob[1])+"\n")

    f = open('OverSampled_test_logreg.txt', 'w+')
    for prob in test_probabilities:
        f.write(str(prob[0])+","+str(prob[1])+"\n")
        
else:
    f = open('UnderSampled_train_logreg.txt', 'w+')
    for prob in training_probabilities:
        f.write(str(prob[0])+","+str(prob[1])+"\n")

    f = open('UnderSampled_test_logreg.txt', 'w+')
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