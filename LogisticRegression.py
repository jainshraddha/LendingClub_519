import numpy as np
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
np.set_printoptions(threshold=np.nan)
from sklearn import preprocessing


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

print Xtest_standardize.shape
y = y.astype('int')

C = [0.1,0.01,0.001,0.0001,1]
clf = LogisticRegressionCV(Cs=C, cv=10, n_jobs=-1, scoring='accuracy')
clf.fit(np.asarray(X_standardize), y)
print "best parameter"
print clf.C_
print "\nTraining Accuracy Score"
print clf.score(X_standardize, y)
print "\n"
predictions = clf.predict_proba(Xtest_standardize)
print type(predictions)


# result1 = ["" for x in range(predictions.shae[0])]
result = 'P(Fully Paid), P(Charged Off)\n'
for i in range(0, predictions.shape[0]):
    value = predictions[i,0]
    val = predictions[i,1]
    result += str(value)+ "," + str(val) + "\n"


f = open('probability_result.csv', 'w')
f.write(result)
f.close()