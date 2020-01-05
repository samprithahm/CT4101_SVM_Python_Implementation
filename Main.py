# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 22:48:47 2019

@author: Sampritha H M
"""


import pandas as pd
import numpy as np
from OneVsRest import OneVsRest
import matplotlib.pyplot as pt

# read CSV
hazelnut = pd.read_csv("hazelnut.csv")
# drop sample_id column
hazelnut = hazelnut.drop("sample_id", axis = 1)

scores = []
# number of iteration
folds = 10
model = OneVsRest()
for i in range(folds):
    print("\n\nRun : ",i+1,"\n\n")
    # data split into train and test
    model.trainTestSplit(hazelnut, 0.66)
    # training the model
    model.trainModel()
    # run test to check the accuracy of the model
    scores.append(model.testModel())

model.plotMetrics()
print("Scores : ",scores)
SVM_Accuracy = float(np.mean(scores)) * int(100)
print("Accuracy of the Model: %0.2f" %SVM_Accuracy, "%")


""" SKLEARN SVM """


print("\n\nComparing accuracy with sklearn SVM\n")
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

hazelnut = pd.read_csv('./hazelnut.csv')
X = hazelnut.drop(['variety','sample_id'],axis=1)
X = preprocessing.scale(X)
y = hazelnut['variety'].values

clf = SVC(kernel='linear', C=1)
cv_scores = cross_val_score(clf, X, y, cv=10)

print("Cross Validation scores: \n",cv_scores)
mean = cv_scores.mean()
mean = float(mean) * int(100)
print("Accuracy of sklearn SVM: %0.2f" %mean,"%")

cv_range = (1,2,3,4,5,6,7,8,9,10)

pt.figure()
pt.plot(cv_range,scores)
pt.plot(cv_range,cv_scores)
pt.xlabel('Number of folds')
pt.ylabel('Accuracy of the model')
model = "OWN IMPLEMENTED SVM","SKLEARN SVM"
pt.legend(model)
pt.savefig("Comaprision.pdf")