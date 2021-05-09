# -*- coding: utf-8 -*-
#!/usr/bin/python3
from __future__ import division

import numpy as np
import numpy.random as rand
import random
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

import pymp

def RedolRegressorException(Exception):
    pass


class RedolRegressor:

    def __init__(self, n_estimators=100, method="regular", perc=0.75, repetitions=3, nearest_neighbours=5, bootstrap=1.0, iterations=3, n_jobs=8):
        self.n_estimators = n_estimators
        self.method = method
        self.perc = perc
        self.repetitions = repetitions
        self.bootstrap = bootstrap
        self.iterations = iterations
        self.n_jobs = n_jobs
        self.nearest_neighbours = nearest_neighbours

    # def get_params(self, deep=True):
    #     # TODO: With deep true it should return base learners params too
    #     return {
    #         'n_estimators': self.n_estimators,
    #         'method': self.method,
    #         'perc': self.perc,
    #         'bootstrap': self.bootstrap,
    #         'n_jobs': self.n_jobs,
    #     }

    # def set_params(self, **params):
    #     valid_params = self.get_params(deep=True)
    #     nested_params = defaultdict(dict)  # grouped by prefix
    #     for key, value in params.items():
    #         key, delim, sub_key = key.partition('__')
    #         if key not in valid_params:
    #             raise ValueError('Invalid parameter %s for estimator %s. '
    #                              'Check the list of available parameters '
    #                              'with `estimator.get_params().keys()`.' %
    #                              (key, self))

    #         if delim:
    #             nested_params[key][sub_key] = value
    #         else:
    #             setattr(self, key, value)
    #             valid_params[key] = value

    #     for key, sub_params in nested_params.items():
    #         valid_params[key].set_params(**sub_params)

    #     return self

    def fit(self, x, y):
        """
        This method is used to fit each one of the decision trees the random noise classifier is composed with.
        This is the way to fit the complete classifier and it is compulsory to carry on with the data classification.

        :param x: original features from the dataset
        :param y: original classes from the dataset
        """
        self.classes = np.unique(y)

        # self.enc = OneHotEncoder(handle_unknown='ignore')
        # self.enc.fit(y.reshape(-1, 1))

        # self.classifiers = []
        self.classifiers = pymp.shared.list()

        with pymp.Parallel(self.n_jobs) as p:
            for n_classifier in p.range(0, self.n_estimators):
                clf = DecisionTreeRegressor()

                _x = x
                _y = y

                # Bootstrap extractions to train the model.
                # These extractions can repeat some indices.
                number_of_extractions = int(self.bootstrap * x.shape[0])
                ind = np.random.randint(0, x.shape[0], number_of_extractions)

                _x = x[ind, :]
                _y = y[ind]

                if self.method == "regular":
                    modified_x, modified_y = self._change_class(_x, _y)
                # elif self.method == "distributed":
                #     modified_x, modified_y = self._change_class_distributed(_x, _y)
                # elif self.method == "randomized":
                #     modified_x, modified_y = self._change_class_randomized(_x, _y)
                else:
                    err_msg = f'The method {self.method} is not a valid method: regular, distributed, randomized'
                    raise RedolRegressorException(err_msg)

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
        This method calculates the probability that a data is well classified or not. It adds a new feature
        to the dataset depending on the suggested_class attribute.

        :param x: data to be classified
        :param suggested_class: new feature to be added
        :return: probabilities that a data is well classified or not
        """
        predictions = []

        for it in range(self.iterations):
            preds = []

            sugg_class = rand.choice(self.classes, size=x.shape[0], replace=True)

            _x = x.copy()
            _x = np.c_[_x, sugg_class]

            # As the method is predicting 0 value I sum the sugg class, not sure about this at all
            [preds.append(clf.predict(_x) + sugg_class) for clf in self.classifiers]
            # preds = np.array(preds).mean(axis=0)
            predictions.append(np.array(preds).T)

        return np.array(predictions).mean(axis=2).mean(axis=0)

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
        random.shuffle(random_data)

        if self.nearest_neighbours:
            updated_data, updated_class = self._regression_NN_change(percentage, random_data, updated_data, y)
        else:
            updated_data, updated_class = self._regression_change(percentage, random_data, updated_data, y)

        return updated_data, np.array(updated_class)

    def _regression_change(self, percentage, random_data, updated_data, y):
        # TODO: This set might be too large as it is saving all y regression values
        # and it s been copied for each num in the loop
        classes = list(set(y))

        new_data = np.array([], dtype=np.int64).reshape(0, updated_data.shape[1])
        new_class = []

        for num in random_data[:percentage]:
            prev_class = updated_data[num, -1]
            classes_without_prev_class = classes.copy()  # copy classes list
            classes_without_prev_class.remove(prev_class)

            # Repeat each example repetitions times and randomize it
            updated_data_repeated = np.tile(updated_data[num, :], (self.repetitions, 1))
            updated_data_repeated[:, -1] = rand.choice(classes_without_prev_class, self.repetitions)
            y_repeated = np.tile(y[num], self.repetitions)
            new_data = np.concatenate((new_data, updated_data_repeated))
            new_class = np.append(new_class, y_repeated)
        
        # Remove randomized data
        old_data = np.delete(updated_data, random_data[:percentage], axis=0)
        old_class = np.delete(y, random_data[:percentage])

        updated_data = np.concatenate((old_data, new_data))
        updated_class = np.append(old_class, new_class)

        # Main difference with RedolClassifier is the new class is the difference
        # between the prev class and another randomly suggested class
        updated_class = (updated_data[:, -1] - updated_class)

        return updated_data, updated_class

    def _regression_NN_change(self, percentage, random_data, updated_data, y):
        # TODO: This set might be too large as it is saving all y regression values
        # and it s been copied for each num in the loop
        classes = list(set(y))

        nn = NearestNeighbors()
        nn.fit(updated_data)

        new_data = np.array([], dtype=np.int64).reshape(0, updated_data.shape[1])
        new_class = []

        for num in random_data[:percentage]:
            prev_class = updated_data[num, -1]
            nn_idx = nn.kneighbors(X=[updated_data[num]], n_neighbors=self.nearest_neighbours, return_distance=False)
            classes_without_prev_class = updated_data[nn_idx, -1]
            classes_without_prev_class = set(classes_without_prev_class[0])
            classes_without_prev_class.discard(prev_class)
            # If classes from neighbours are equal, we choose randomly from any other class
            if not classes_without_prev_class:
                classes_without_prev_class = classes.copy()
                classes_without_prev_class.remove(prev_class)

            # Repeat each example repetitions times and randomize it
            updated_data_repeated = np.tile(updated_data[num, :], (self.repetitions, 1))
            updated_data_repeated[:, -1] = rand.choice(list(classes_without_prev_class), self.repetitions)
            y_repeated = np.tile(y[num], self.repetitions)
            new_data = np.concatenate((new_data, updated_data_repeated))
            new_class = np.append(new_class, y_repeated)
        
        # Remove randomized data
        old_data = np.delete(updated_data, random_data[:percentage], axis=0)
        old_class = np.delete(y, random_data[:percentage])

        updated_data = np.concatenate((old_data, new_data))
        updated_class = np.append(old_class, new_class)

        # Main difference with RedolClassifier is the new class is the difference
        # between the prev class and another randomly suggested class
        updated_class = (updated_data[:, -1] - updated_class)

        return updated_data, updated_class
