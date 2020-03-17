#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:43:23 2020

@author: nk
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

class ai:

    def __init__(self, data, features, target, test_size, model, params):
        self.X = data.loc[:, features].values
        self.y = data[target].values.ravel()
        self.clf = self.learn(self.X, self.y, test_size, model, params)

    def learn(self, X, y, test_size, model, params=None):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)

        if model == "lr":
            if params == None:
                clf = LogisticRegression()
            else:
                clf = LogisticRegression(**params["lr"])
            clf.fit(X_train, y_train)
        elif model == "dt":
            if params == None:
                clf = DecisionTreeClassifier()
            else:
                clf = DecisionTreeClassifier(**params["dt"])
            clf.fit(X_train, y_train)
        else:
            print("Invalid Model")
            clf = None
        return clf