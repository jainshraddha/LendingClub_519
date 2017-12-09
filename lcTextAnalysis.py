
import numpy as np
import time as time
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from matplotlib import colors as mcolors
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict

if __name__ == '__main__':

  filename = "textTrainingFinal.csv"

  raw_data = pd.read_csv(filename, sep=',', names = ['id', 'member_id', 'int_rate', 'installment',
   'grade', 'sub_grade', 'desc', 'loan_status', 'status_code'], encoding='latin-1')

 
  training_data = raw_data["desc"].values.astype('U')
  training_target = raw_data["status_code"].values.astype('int32')

  training_data= training_data[1:]
  training_target= training_target[1:]

  print(len(training_data))
  print(len(training_target))
  print(training_data[0])
  print(training_target[0])

  over_sampling = False
  under_sampling = False

  classifier_name = 'MNB'

  if over_sampling:
    classifier_name = 'MNB with Oversampling'
    ros = RandomOverSampler(random_state=0)
    training_data, training_target = ros.fit_sample(training_data.reshape(-1,1), training_target.reshape(-1,1))
    training_data = [training_data.item(i) for i in range(len(training_data))]
  elif under_sampling:
    classifier_name = 'MNB with Undersampling'
    rus = RandomUnderSampler(random_state=0)
    training_data, training_target = rus.fit_sample(training_data.reshape(-1,1), training_target.reshape(-1,1))
    training_data = [training_data.item(i) for i in range(len(training_data))]

  mnb_clf = Pipeline([('vect', CountVectorizer(stop_words='english',lowercase=True,ngram_range=(1,1))), 
                    ('tfidf', TfidfTransformer(sublinear_tf=True)),
                    ('clf', MultinomialNB(alpha=1e-1)),])

  scoring = ['accuracy', 'roc_auc']
  scores = cross_validate(mnb_clf, training_data, training_target, scoring=scoring,cv=10, return_train_score=True)
  train_accuracy = np.mean(scores['train_accuracy'])
  test_accuracy = np.mean(scores['test_accuracy'])
  train_auc = np.mean(scores['train_roc_auc'])
  test_auc = np.mean(scores['test_roc_auc'])
  runtime = np.mean(scores['score_time'])

  print("-----------------------------------------------------")
  print("      "+classifier_name)
  print("Train/Test Accuracy : " + str(train_accuracy)+"/"+str(test_accuracy))
  print("Train/Test AUC : " + str(train_auc)+"/"+str(test_auc))
  print("Training Time "+str(runtime)+" seconds")
  print("-----------------------------------------------------")

  training_probabilities = cross_val_predict(mnb_clf, training_data, training_target, cv=10, method='predict_proba')
  if over_sampling:
    f = open('probsMNBOverTest.txt', 'w')
  elif under_sampling:
    f = open('probsMNBUnderTest.txt', 'w')
  else:
    f = open('probsMNBTest.txt', 'w')
  for prob in training_probabilities:
    f.write(str(prob[0])+","+str(prob[1])+"\n")

  f.close()



  
  



  
