# -*- coding: utf-8 -*-
#!/usr/bin/python3
from __future__ import division

import numpy as np
from random import *
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

import pymp


class RedolClassifier:

    def __init__(self, n_estimators=100, perc=0.75, bagg=True, classifier='tree', n_jobs=1):
        self.n_estimators = n_estimators
        self.perc = perc
        self.bagg = bagg
        self.classifier = classifier
        self.n_jobs = n_jobs

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

        # self.classifiers = []
        self.classifiers = pymp.shared.list()

        with pymp.Parallel(self.n_jobs) as p:
            for n_classifier in p.range(0, self.n_estimators):
                if self.classifier == 'tree':
                    clf = tree.DecisionTreeClassifier()
                elif self.classifier == 'regression':
                    clf = LogisticRegression()

                _x = x
                _y = y

                if self.bagg:
                    ind = np.random.randint(0, x.shape[0], x.shape[0])

                    _x = x[ind, :]
                    _y = y[ind]

                modified_x, modified_y = self._change_class(_x, _y)

                clf.fit(modified_x, modified_y)

                # self.classifiers[n_classifier] = clf
                with p.lock:
                    self.classifiers.append(clf)

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
            if len(self.classes) > 2:
                _x = np.c_[_x, np.tile(self.enc.transform(cl.reshape(-1, 1)).toarray(), (x.shape[0],1))]
            else:
                _x = np.c_[_x, np.repeat(cl, x.shape[0])]

            [preds.append(clf.predict_proba(_x)) for clf in self.classifiers]
            preds = np.array(preds).mean(axis=0)
            predictions.append(preds[:, 1])

        return np.array(predictions).transpose()

    def predict_proba_error(self, x):
        """
        This method calculates a matrix which contains the probabilities of each example cumulatively.

        :param x: the original features from the dataset
        :param suggested_class: the class the classifier uses as new feature
        :return: the final probabilities matrix
        """
        self.predictions = []

        for cl in self.classes:
            preds = []

            _x = x.copy()
            if len(self.classes) > 2:
                _x = np.c_[_x, np.tile(self.enc.transform(cl.reshape(-1, 1)).toarray(), (x.shape[0],1))]
            else:
                _x = np.c_[_x, np.repeat(cl, x.shape[0])]

            [preds.append(clf.predict_proba(_x)) for clf in self.classifiers]
            preds = np.array(preds)

            for i in range(len(self.classifiers)-1, -1, -1):
                preds[i, :, :] = preds[:i+1, :, :].sum(axis=0)
                preds[i, :, :] /= i+1

            self.predictions.append(preds[:, :, 1].transpose())

        self.predictions = np.array(self.predictions).transpose()

        return self.predictions

    def score_error(self, x, y, n_classifiers=100):
        """
        With this method we are able to see what is going on with the classification of the examples for each classifier.
        This method allows us to calculate the score obtained using the amount of classifiers we want up to the maximum
        of classifiers with which it was declared.

        :param x: original features dataset
        :param y: original classes from the dataset
        :param n_classifiers: number of classifiers used to calculate the score
        :return: score obtained
        """
        if n_classifiers is None:
            n_classifiers = len(self.classifiers)

        n_classifiers -= 1

        return sum([1 for i, pred in enumerate(self.predictions[n_classifiers, :, :]) if float(np.where(pred == np.amax(pred))[0][0] == y[i])]) / x.shape[0]

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

        percentage = int(num_data * self.perc)

        updated_data = data.copy()

        random_data = list(range(0, num_data))
        shuffle(random_data)

        if len(self.classes) <= 2:
            updated_data = self._binary(percentage, random_data, updated_data)
        else:
            updated_data = self._multiclass(percentage, random_data, updated_data, y)

        updated_class = [(updated_data[i, -1] == data[i, -1]) for i in range(0, num_data)]

        # Changes the old class from the data features to the one hot encoder ones
        if len(self.classes) > 2:
            updated_data = np.c_[updated_data[:,:-1], self.enc.transform(updated_data[:, -1].reshape(-1, 1)).toarray()]

        return updated_data, np.array(updated_class)

    def _binary(self, percentage, random_data, updated_data):
        for num in random_data[:percentage]:
            updated_data[num, -1] = 1 - updated_data[num, -1]

        return updated_data

    def _multiclass(self, percentage, random_data, updated_data, y):
        classes = list(set(y))

        for num in random_data[:percentage]:
            prev_class = updated_data[num, -1]
            classes_without_prev_class = classes.copy()  # copy classes list
            classes_without_prev_class.remove(prev_class)
            updated_data[num, -1] = choice(classes_without_prev_class)

        return updated_data
