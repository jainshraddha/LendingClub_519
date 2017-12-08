
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


def printMetrics(classifier_name,train_predictions,train_observed,test_predictions,test_observed,runtime,include_test):


  if include_test:
    train_metrics_list = metrics.precision_recall_fscore_support(train_observed, train_predictions,average='weighted')
    train_precision = round(train_metrics_list[0],4)
    train_recall = round(train_metrics_list[1],4)
    train_accuracy = round(np.mean(train_predictions == train_observed),4)

    test_metrics_list = metrics.precision_recall_fscore_support(test_observed, test_predictions,average='weighted')
    test_precision = round(test_metrics_list[0],4)
    test_recall = round(test_metrics_list[1],4)
    test_accuracy = round(np.mean(test_predictions == test_observed),4)

    print("-----------------------------------------------------")
    print("      "+classifier_name)
    print("Train/Test Accuracy : " + str(train_accuracy)+"/"+str(test_accuracy))
    print("Train/Test Precision : " + str(train_precision)+"/"+str(test_precision))
    print("Train/Test Recall : " + str(train_recall)+"/"+str(test_recall))
    print("Training Time "+str(runtime)+" seconds")
    print("-----------------------------------------------------")
  else:
    train_metrics_list = metrics.precision_recall_fscore_support(train_observed, train_predictions,average='weighted')
    train_precision = round(train_metrics_list[0],4)
    train_recall = round(train_metrics_list[1],4)
    train_accuracy = round(np.mean(train_predictions == train_observed),4)


    print("-----------------------------------------------------")
    print("      "+classifier_name)
    print("Train Accuracy : " + str(train_accuracy))
    print("Train Precision : " + str(train_precision))
    print("Train Recall : " + str(train_recall))
    print("Training Time "+str(runtime)+" seconds")
    print("-----------------------------------------------------")


def trainMNB(include_test):
  mnb_clf = Pipeline([('vect', CountVectorizer(stop_words='english',lowercase=True,ngram_range=(1,1))), 
                    ('tfidf', TfidfTransformer(sublinear_tf=True)),
                    ('clf', MultinomialNB(alpha=1e-1)),])
                    # ('clf', MultinomialNB()),]) 

  if include_test:
    start = time.clock()
    optimal_mnb_clf = mnb_clf.fit(training_data, training_target)
    runtime = round(time.clock()-start,4)
    train_predictions = optimal_mnb_clf.predict(training_data)
    test_predictions = optimal_mnb_clf.predict(test_data)
    printMetrics("MultinomialNB",train_predictions,training_target,test_predictions,test_target,runtime,include_test)
    return optimal_mnb_clf
   

  else:
    start = time.clock()
    optimal_mnb_clf = mnb_clf.fit(training_data, training_target)
    runtime = round(time.clock()-start,4)
    train_predictions = optimal_mnb_clf.predict(training_data)
    printMetrics("MultinomialNB",train_predictions,training_target,None,None,runtime,include_test)
    return optimal_mnb_clf

def trainCos(include_test):
  cos_clf = Pipeline([('vect', CountVectorizer(stop_words='english',lowercase=True)), 
                    ('tfidf', TfidfTransformer(sublinear_tf=True)), 
                    ('clf', svm.SVC(kernel='linear',C=5,probability=True)),])
  if include_test:
    start = time.clock()
    optimal_cos_clf = cos_clf.fit(training_data, training_target)
    runtime = round(time.clock()-start,4)
    train_predictions = optimal_cos_clf.predict(training_data)
    test_predictions = optimal_cos_clf.predict(test_data)
    printMetrics("SVM Cosine Similarity Kernel",train_predictions,training_target,test_predictions,test_target,runtime,include_test)
  # printMetrics("SVM Cosine Similarity Kernel",train_predictions,training_target,None,None,runtime)
  else:
    start = time.clock()
    optimal_cos_clf = cos_clf.fit(training_data, training_target)
    runtime = round(time.clock()-start,4)
    train_predictions = optimal_cos_clf.predict(training_data)
    printMetrics("SVM Cosine Similarity Kernel",train_predictions,training_target,None,None,runtime,include_test)
    training_probabilities = optimal_cos_clf.predict_proba(training_data)
    return training_probabilities

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

  include_test = True

  test_percentage = 0.20

  if (include_test):
    training_data, test_data, training_target, test_target = train_test_split(training_data, training_target, test_size=test_percentage,random_state=0)
    
    over_sampling = False
    under_sampling = True

    print("Training MNB")

    full_training_data = training_data
    
    if over_sampling:
      ros = RandomOverSampler(random_state=0)
      training_data, training_target = ros.fit_sample(training_data.reshape(-1,1), training_target.reshape(-1,1))
      training_data = [training_data.item(i) for i in range(len(training_data))]
    elif under_sampling:
      rus = RandomUnderSampler(random_state=0)
      training_data, training_target = rus.fit_sample(training_data.reshape(-1,1), training_target.reshape(-1,1))
      training_data = [training_data.item(i) for i in range(len(training_data))]
    
    optimal_mnb_clf = trainMNB(include_test)
    training_probabilities = optimal_mnb_clf.predict_proba(training_data)
    test_probabilities = optimal_mnb_clf.predict_proba(test_data)
    print("Training AUC: {}".format(roc_auc_score(training_target, training_probabilities[:,1])))
    print("Test AUC: {}".format(roc_auc_score(test_target, test_probabilities[:,1])))

    # if over_sampling or under_sampling:
    training_probabilities = optimal_mnb_clf.predict_proba(full_training_data)

    if over_sampling:
      f = open('probsMNBOverTest.txt', 'w')
    elif under_sampling:
      f = open('probsMNBUnderTest.txt', 'w')
    else:
      f = open('probsMNBTest.txt', 'w')
    for prob in training_probabilities:
      f.write(str(prob[0])+","+str(prob[1])+"\n")

    f.close()

  else:
    # plotROC()

    over_sampling = True
    under_sampling = False

    print("Training MNB")


    if over_sampling:
      full_training_data = training_data
      ros = RandomOverSampler(random_state=0)
      training_data, training_target = ros.fit_sample(training_data.reshape(-1,1), training_target.reshape(-1,1))
      training_data = [training_data.item(i) for i in range(len(training_data))]
    elif under_sampling:
      full_training_data = training_data
      rus = RandomUnderSampler(random_state=0)
      training_data, training_target = rus.fit_sample(training_data.reshape(-1,1), training_target.reshape(-1,1))
      training_data = [training_data.item(i) for i in range(len(training_data))]
    
    optimal_mnb_clf = trainMNB(include_test)
    training_probabilities = optimal_mnb_clf.predict_proba(training_data)
    print("Training AUC: {}".format(roc_auc_score(training_target, training_probabilities[:,1])))

    # if over_sampling or under_sampling:
    training_probabilities = optimal_mnb_clf.predict_proba(full_training_data)

    if over_sampling:
      f = open('probsMNBOver.txt', 'w')
    elif under_sampling:
      f = open('probsMNBUnder.txt', 'w')
    else:
      f = open('probsMNB.txt', 'w')
    for prob in training_probabilities:
      f.write(str(prob[0])+","+str(prob[1])+"\n")

    f.close()

    # print("Training Cos")
    # training_probabilities = trainCos(include_test)

    # f = open('probsCos.txt', 'w')
    # for prob in training_probabilities:
    #   f.write(str(prob[0])+","+str(prob[1])+"\n")

    # f.close()



  
  



  