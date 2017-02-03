#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

print( "classification start..." )


#########################################################
### your code goes here ###

#########################################################

from sklearn.svm import SVC

# linear kernel
#clf = SVC(kernel='linear')

# rbf kernel
#clf = SVC(kernel='rbf', C=1.)
#clf = SVC(kernel='rbf', C=10.)
#clf = SVC(kernel='rbf', C=100.)
#clf = SVC(kernel='rbf', C=1000.)
clf = SVC(kernel='rbf', C=10000.)

# slice inputs by 100
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

clf.fit(features_train, labels_train)

print("fitting complete. prediction start")

pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(labels_test, pred)
print("accuracy: " + str(accuracy))

num_class_one = 0
for i in range(len(pred)):
    if pred[i] == 1:
        num_class_one += 1

print(num_class_one)