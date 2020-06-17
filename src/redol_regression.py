# -*- coding: utf-8 -*-
#!/usr/bin/python3
from __future__ import division

import numpy as np
from random import *
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


class RegressionRedol:

    def __init__(self, n_estimators=100, perc=0.75, bagg=True):
        self.n_estimators = n_estimators
        self.perc = perc
        self.bagg = bagg

    def fit(self, x, y):
        """
        This method is used to fit each one of the decision trees the random noise classifier is composed with.
        This is the way to fit the complete classifier and it is compulsory to carry on with the data classification.

        :param x: original features from the dataset
        :param y: original classes from the dataset
        """
        self.classes = np.unique(y)

        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.enc.fit(y.reshape(-1, 1))

        modified_x, modified_y = self._change_class(x, y)

        self.clf = LogisticRegression()
        self.clf.fit(modified_x, modified_y)

    def score(self, x, y):
        """
        This method is used to calculate the classifier accuracy comparing the obtained classes with the original
        ones from the dataset.

        :param x: original features from the dataset
        :param y: original classes from the dataset
        :param suggested_class: a new feature added to classify the examples
        :return: classifier accuracy
        """
        return sum([1 for i, prediction in enumerate(self.predict(x)) if prediction == y[i]])/x.shape[0]

    def predict(self, x):
        """
        This method is used to generate the class predictions from each example to be classified.
        It uses the method predict_proba to calculate the probabilities that a data is well classified or not.

        :param x: original features from the dataset
        :param suggested_class: a new feature added to classify the examples
        :return: an array with the predicted class for each example from the dataset
        """
        return np.array([float(np.where(pred == np.amax(pred))[0][0]) for pred in self.predict_proba(x)])

    def predict_proba(self, x):
        """
        This method calculates the probability that a data is well classified or not. It adds a new feature
        to the dataset depending on the suggested_class attribute.

        :param x: data to be classified
        :param suggested_class: new feature to be added
        :return: probabilities that a data is well classified or not
        """
        predictions = []

        for cl in self.classes:
            preds = []

            _x = x.copy()
            _x = np.c_[_x, np.tile(self.enc.transform(cl.reshape(-1, 1)).toarray(), (x.shape[0],1))]

            preds = self.clf.predict_proba(_x)
            predictions.append(preds[:, 1])

        return np.array(predictions).transpose()

    def _change_class(self, x, y):
        """
        Given a data set split in features and classes this method transforms this set into another set.
        This new set is created based on random noise generation and its classification. The randomization
        of the new data set is given by the percentage received from the class constructor.

        The randomization is generated changing some data classes and pointing it out in the new class.
        The new class is calculated comparing the original class with the data randomization. For this new class
        '1' means "well classified" and '0', the opposite.

        :param x: features from the original data set
        :param y: classes from the original data set
        :return: features and classes from the new data set
        """
        data = np.c_[x, y]

        num_data = data.shape[0]

        updated_data = []

        for _instance in data:
            for _cl in self.classes:
                _new_class = 1. if _instance[-1] == _cl else 0.
                _new_instance = np.concatenate([_instance[:-1], self.enc.transform([[_cl]]).toarray()[0]])
                _new_instance = np.append(_new_instance, _new_class)

                updated_data.append(_new_instance)

        updated_data = np.array(updated_data)

        return updated_data[:,:-1], updated_data[:,-1]
