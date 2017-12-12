import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
import pandas as pd
np.set_printoptions(threshold=np.nan)
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

###########----------TRAINING DATA---------------###########
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

y = y.astype('int')


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

Xtest_standardize = Xtest[:, 0:15]
Xtest = Xtest[:,15:]
Xtest_standardize = preprocessing.scale(Xtest_standardize)
Xtest_standardize = np.concatenate((Xtest_standardize, Xtest), axis=1)

# for i in range (0, 10):
over_sampling = False
under_sampling = True

# training_data, test_data, training_target, test_target = train_test_split(X_standardize, y, test_size=0.20, random_state=42)

training_data = X_standardize
training_target = y

print "training data"
print training_data.shape

classifier_name = 'Logistic Regression'

if over_sampling:
    classifier_name = 'Logistic Regression with Oversampling'
    print classifier_name
    ros = RandomOverSampler(random_state=0)
    training_data, training_target = ros.fit_sample(training_data, training_target)

elif under_sampling:
    classifier_name = 'Logistic Regression with Undersampling'
    print classifier_name
    rus = RandomUnderSampler(random_state=0)
    training_data, training_target = rus.fit_sample(training_data, training_target)

C = [0.1,0.01,0.001,0.0001,1]


clf = LogisticRegressionCV(Cs=C, cv=10, n_jobs=-1, scoring='accuracy')
clf.fit(training_data, training_target)
print "best parameter"
print np.asscalar(clf.C_)

clf_model = LogisticRegression(C=np.asscalar(clf.C_))

print "\nTraining Accuracy Score"
print clf.score(training_data, training_target)

scoring = ['accuracy', 'roc_auc']
scores = cross_validate(clf_model, training_data, training_target, scoring=scoring, cv=10, return_train_score=True)
train_accuracy = np.mean(scores['train_accuracy'])
test_accuracy = np.mean(scores['test_accuracy'])
train_auc = np.mean(scores['train_roc_auc'])
test_auc = np.mean(scores['test_roc_auc'])
runtime = np.mean(scores['score_time'])


print("-----------------------------------------------------")
print("      " + classifier_name)
print("Train/Test Accuracy : " + str(train_accuracy) + "/" + str(test_accuracy))
print("Train/Test AUC : " + str(train_auc) + "/" + str(test_auc))
print("Training Time " + str(runtime) + " seconds")
print("-----------------------------------------------------")

clf_model.fit(training_data, training_target)
training_probabilities = clf_model.predict_proba(Xtest_standardize)
if over_sampling:
    f = open('probsLogisticOverTest.txt', 'w')
elif under_sampling:
    f = open('probsLogisticUnderTest.txt', 'w')
else:
    f = open('probsLogisticTest.txt', 'w')
for prob in training_probabilities:
    f.write(str(prob[0]) + "," + str(prob[1]) + "\n")

f.close()

