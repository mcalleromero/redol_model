import sys
sys.path.append('/home/mario.calle/master/redol_model/')

import time

import util.properties as properties

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

from redol.redol import *


def get_data():
    if len(sys.argv) > 1:
        try:
            dataset = pd.read_csv(properties.DATASET_DIRECTORY + sys.argv[1])

            cat_columns = ['class']

            if sys.argv[1] == "tic-tac-toe":
                cat_columns = dataset.select_dtypes(['object']).columns

            dataset[cat_columns] = dataset[cat_columns].astype('category')
            dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes)
            # xdataset = pd.get_dummies(dataset, columns=cat_columns)
            dataset = dataset.values

            X, y = dataset[:,:-1], dataset[:,-1]

            return X, y
        except IOError:
            print("File \"{}\" does not exist.".format(sys.argv[1]))
            return

    else:
        raise ValueError("File not found")

def main():

    X, y = get_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    n_trees = 100

    redolclf = RedolClassifier(n_estimators=n_trees, perc=0.75, n_jobs=8)
    rfclf = RandomForestClassifier(n_estimators=n_trees)
    boostingclf = GradientBoostingClassifier(n_estimators=n_trees)
    baggingclf = BaggingClassifier(n_estimators=n_trees)

    starttime = time.time()

    print("Entrenamiento Redol\n")
    redolclf.fit(X_train, y_train)
    redolclf.predict(X_test)

    print('That took {} seconds'.format(time.time() - starttime))

    starttime = time.time()

    print("Entrenamiento RF\n")
    rfclf.fit(X_train, y_train)
    rfclf.predict(X_test)

    print('That took {} seconds'.format(time.time() - starttime))

    starttime = time.time()

    print("Entrenamiento Boosting\n")
    boostingclf.fit(X_train, y_train)
    boostingclf.predict(X_test)

    print('That took {} seconds'.format(time.time() - starttime))

    starttime = time.time()

    print("Entrenamiento Bagging\n")
    baggingclf.fit(X_train, y_train)
    baggingclf.predict(X_test)

    print('That took {} seconds'.format(time.time() - starttime))

    print("----------------------------------------------")
    print("{} Redol:{} {}".format(properties.COLOR_BLUE, properties.END_C, redolclf.score(X_test, y_test)))
    print("{} Random forest score:{} {}".format(properties.COLOR_BLUE, properties.END_C, rfclf.score(X_test, y_test)))
    print("{} Boosting score:{} {}".format(properties.COLOR_BLUE, properties.END_C, boostingclf.score(X_test, y_test)))
    print("{} Bagging score:{} {}".format(properties.COLOR_BLUE, properties.END_C, baggingclf.score(X_test, y_test)))


if __name__ == "__main__":
    main()
