import sys
sys.path.append('/home/cromero/projects/redol_model/')

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

import time
import util.properties as properties

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

from redol import RedolClassifier

def get_data(model):
    if model:
        try:
            dataset = pd.read_csv(f'../data/original/{model}.csv')

            cat_columns = ['class']

            if model == "tic-tac-toe":
                cat_columns = dataset.select_dtypes(['object']).columns

            dataset[cat_columns] = dataset[cat_columns].astype('category')
            dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes)
            # xdataset = pd.get_dummies(dataset, columns=cat_columns)
            dataset = dataset.values

            X, y = dataset[:,:-1], dataset[:,-1]

            return X, y
        except IOError:
            print("File \"{}\" does not exist.".format(model))
            return
    else:
        raise ValueError("File not found")

def main():

    models = ["australian", "diabetes", "german", "heart", "ionosphere", "magic04", "new-thyroid", "ringnorm", "segment", "threenorm", "tic-tac-toe", "twonorm", "waveform", "wdbc", "wine"]

    n_trees = 100

    regs = []
    dist = []
    randomized = []

    for model in models:
        X, y = get_data(model)

        regularclf = RedolClassifier(n_estimators=n_trees, method="regular", perc=0.5, n_jobs=8)
        distributedclf = RedolClassifier(n_estimators=n_trees, method="distributed", perc=0.5, n_jobs=8)
        randomizedclf = RedolClassifier(n_estimators=n_trees, method="randomized", perc=0.5, n_jobs=8)

        regular_acc = []
        distributed_acc = []
        randomized_acc = []

        n_splits = 10
        skf = StratifiedKFold(n_splits=n_splits)

        starttime = time.time()

        i = 0

        for train_index, test_index in skf.split(X, y):
            i += 1
            print(f"ITERACION {i}/{n_splits}")

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            starttime = time.time()

            print("Entrenamiento REGULAR\n")
            regularclf.fit(X_train, y_train)
            regular_y = regularclf.predict(X_test)
            regular_acc.append(1 - metrics.accuracy_score(y_test, regular_y))

            print('That took {} seconds'.format(time.time() - starttime))

            print("Entrenamiento DISTRIBUTED\n")
            distributedclf.fit(X_train, y_train)
            distributed_y = distributedclf.predict(X_test)
            distributed_acc.append(1 - metrics.accuracy_score(y_test, distributed_y))

            print('That took {} seconds'.format(time.time() - starttime))

            print("Entrenamiento RANDOMIZED\n")
            randomizedclf.fit(X_train, y_train)
            randomized_y = randomizedclf.predict(X_test)
            randomized_acc.append(1 - metrics.accuracy_score(y_test, randomized_y))

            print('That took {} seconds'.format(time.time() - starttime))

        print(f"{model}")
        print("----------------------------------------------")
        print("{} REGULAR err:{} {}".format(properties.COLOR_BLUE, properties.END_C, np.mean(regular_acc)))
        print("{} DISTRIBUTED err:{} {}".format(properties.COLOR_BLUE, properties.END_C, np.mean(distributed_acc)))
        print("{} RANDOMIZED err:{} {}".format(properties.COLOR_BLUE, properties.END_C, np.mean(randomized_acc)))
        regs.append(np.mean(regular_acc))
        dist.append(np.mean(distributed_acc))
        randomized.append(np.mean(randomized_acc))

    print(np.mean(regs))
    print(np.mean(dist))
    print(np.mean(randomized))

if __name__ == "__main__":
    main()
