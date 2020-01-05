# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:36:36 2019

@author: Sampritha H M
"""
import numpy as np

class SVM:
    def __init__(self):
        pass

    
    def fit(self):
        # Convert y classes to +1, -1
        bi_y = np.zeros((len(self.y), 1))
        negative_class = self.classes[0]
        positive_class = self.classes[1]
        for i in range(0, len(self.y)):
            if self.y[i] == negative_class:
                bi_y[i] = -1
            elif self.y[i] == positive_class:
                bi_y[i] = 1

        # Add weight to X for stochastic gradient descent
        X_weight = np.empty((len(self.X), 1))
        X_weight.fill(-1)
        weighted_X = np.append(self.X, X_weight, axis=1)
        
        #Initialize our SVMs weight vector with zeros
        weight_vector = np.zeros(len(weighted_X[0]))
        #The learning rate
        LR = 1
        # training iteration
        iterations = 10000

        # implementing Stochastic Gradient Descent
        for iteration in range(1, iterations):
            for i, x in enumerate(self.X):
                # missclassification condition
                if (bi_y[i] * np.dot(weighted_X[i], weight_vector)) < 1:
                    # updating wights when missclassified
                    weight_vector = weight_vector + LR * ((weighted_X[i] * bi_y[i]) + (-2 * (1/iteration) * weight_vector))
                else:
                    # updatig weight when correctly classified
                    weight_vector = weight_vector + LR * (-2 * (1/iteration) * weight_vector)
        self.w = weight_vector[:-1]
        self.b = weight_vector[-1]

    
    def predict(self, feature):
        # predict wheather faeture belogs to a class or not
        result = np.sign(np.dot(self.w, np.array(feature)) - self.b)
        if result == -1:
            # doesnot belong
            return self.classes[0]
        elif result == 1:
            # belongs
            return self.classes[1]
        else: return "Error"
    
    def trainSVM(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.fit()