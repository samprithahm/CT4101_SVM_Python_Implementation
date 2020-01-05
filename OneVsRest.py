# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:33:34 2019

@author: Sampritha H M
"""

import numpy as np
from SVM import SVM
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

class OneVsRest:

    def __init__(self):
        pass
    
    def trainTestSplit(self, data, train_test_split=0.66):
        # read features
        self.data = data.values
        # shuffle data randomly
        np.random.shuffle(self.data)
        self.train_data_len = int(len(self.data) * train_test_split)
        
        # generate training and test data based on split ratio
        self.train_data = self.data[:self.train_data_len]
        self.test_data = self.data[self.train_data_len:]
        
        # generate X and y for train and test data
        self.X_train = self.train_data[:,:-1]
        # Standard scaling
        self.X_train = preprocessing.scale(self.X_train)
        self.y_train = self.train_data[:, -1]

        self.X_test = self.test_data[:,:-1]
        # Standard scaling
        self.X_test = preprocessing.scale(self.X_test)
        self.y_test = self.test_data[:, -1]
        
        # get uniquie class in y_train
        self.classes = np.unique(self.y_train)
        
        # attach svm instances to each of the class identified
        self.classification_object = self.identifyClassifier(self.classes)

    def identifyClassifier(self, classes):
        classification_object = [[]]

        for nut in classes:
            svm = SVM()
            # associate each class to an object of SVM
            classification_object.append([nut, svm])

        classification_object.pop(0)
        return classification_object

    def trainModel(self):
        print("Training each distinct class")
        for nut, svm in self.classification_object:
            X_copy = np.copy(self.X_train)
            y_copy = np.copy(self.y_train)
            
            # add 'not class' to each class to enable binary classification
            y_copy[y_copy != nut] = "Not" + nut
            
            #train the model for each class
            svm.trainSVM(X_copy, y_copy)

            print(nut, "is now trained with SVM")

    def predictClass(self, features):
        prediction = []
        outcome = ""
        # for each class, predict if feature belong to the class using associate svm object
        for nut, svm in self.classification_object:
            prediction.append(svm.predict(features))
        
        for pred in prediction:
            for nut in self.classes:
                if pred == nut:
                    outcome = pred
            if(outcome == ""):
                outcome = self.classes[0]
        return outcome

    def testModel(self):
        row = []
        correct_prediction = 0
        total_predictions = len(self.X_test)
        self.y_pred = []
        self.y_actual = []
        for i in range(total_predictions):
            predicted_class = self.predictClass(self.X_test[i])
            self.y_pred.append(predicted_class)
            actual_class = self.y_test[i]
            self.y_actual.append(actual_class)
            row.append("Predicted Value: %s , Actual Value: %s" % (predicted_class,actual_class))
            if predicted_class == actual_class:
                correct_prediction += 1
            
            with open('hazelnut_prediction_result.csv', 'a') as writer:
                writer.write("\n")
                writer.write("Iteration Running")
                writer.write("\n")
                for line in row:
                    writer.write(line + "\n")
                writer.close()
        accuracy_score = correct_prediction/total_predictions
        
        return accuracy_score
    
    def plotMetrics(self):
        plt.figure()
        cm = confusion_matrix(self.y_actual, self.y_pred, labels=self.classes)
        print(cm)
        sn.heatmap(cm, annot=True, cbar=False)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.pdf')