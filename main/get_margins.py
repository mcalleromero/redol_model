import sys
sys.path.append('/home/mario.calle/master/redol_model/')

import time

import util.properties as properties

from sklearn.model_selection import train_test_split
from sklearn import metrics
from mlxtend.evaluate import bias_variance_decomp

import pandas as pd
import numpy as np

from redol.redol import RedolClassifier
from redol.rf_votings import RandomForestVotings
from redol.boosting_votings import GradientBoostingVotings
from redol.bagging_votings import BaggingVotings
from util import generator

import matplotlib.pyplot as plt


def get_data():
    if len(sys.argv) > 1:
        try:
            dataset = pd.read_csv(f'{properties.DATASET_DIRECTORY}original/{sys.argv[1]}.csv')

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    n_trees = 100

    redolclf = RedolClassifier(n_estimators=n_trees, perc=0.75)
    rfclf = RandomForestVotings(n_estimators=n_trees)
    boostingclf = GradientBoostingVotings(n_estimators=n_trees)
    baggingclf = BaggingVotings(n_estimators=n_trees)

    starttime = time.time()

    print("Entrenamiento Redol\n")
    redolclf.fit(X_train, y_train)
    redol_margins, redol_preds = redolclf.get_margins(X_test, y_test)
    redol_margins_train, redol_preds_train = redolclf.get_margins(X_train, y_train)

    print('That took {} seconds'.format(time.time() - starttime))

    starttime = time.time()

    print("Entrenamiento RF\n")
    rfclf.fit(X_train, y_train)
    margins, preds = rfclf.get_margins(X_test, y_test)
    margins_train, preds_train = rfclf.get_margins(X_train, y_train)

    print('That took {} seconds'.format(time.time() - starttime))

    starttime = time.time()

    print("Entrenamiento BOOSTING\n")
    boostingclf.fit(X_train, y_train)
    boostingmargins, boostingpreds = boostingclf.get_margins(X_test, y_test)
    boostingmargins_train, boostingpreds_train = boostingclf.get_margins(X_train, y_train)

    print('That took {} seconds'.format(time.time() - starttime))

    print("Entrenamiento BAGGING\n")
    baggingclf.fit(X_train, y_train)
    baggingmargins, baggingpreds = baggingclf.get_margins(X_test, y_test)
    baggingmargins_train, baggingpreds_train = baggingclf.get_margins(X_train, y_train)

    print('That took {} seconds'.format(time.time() - starttime))

    # print('MARGINS')
    # print(f'\tREDOL:\t{redol_margins}')
    # print(f'\tR. FOREST:\t{margins}')
    # print(f'\tR. FOREST:\t{boostingmargins}')
    print('\nORIGINAL SCORE')
    print(f'\tREDOL:\t{metrics.accuracy_score(redolclf.predict(X_test), y_test)}')
    print(f'\tR. FOREST:\t{metrics.accuracy_score(rfclf.predict(X_test), y_test)}')
    print(f'\tBOOSTING:\t{metrics.accuracy_score(boostingclf.predict(X_test), y_test)}')
    print(f'\tBAGGING:\t{metrics.accuracy_score(baggingclf.predict(X_test), y_test)}')
    print('\nMARGINS SCORE')
    print(f'\tREDOL:\t{metrics.accuracy_score(redol_preds, y_test)}')
    print(f'\tR. FOREST:\t{metrics.accuracy_score(preds, y_test)}')
    print(f'\ttBOOSTING:\t{metrics.accuracy_score(boostingpreds, y_test)}')
    print(f'\ttBOOSTING:\t{metrics.accuracy_score(baggingpreds, y_test)}')

    plt.plot(np.sort(redol_margins), np.linspace(0, 1, len(redol_margins)))
    plt.plot(np.sort(margins), np.linspace(0, 1, len(redol_margins)))
    # plt.plot(np.sort(boostingmargins), np.linspace(0, 1, len(redol_margins)))
    # plt.plot(np.sort(baggingmargins), np.linspace(0, 1, len(redol_margins)))

    plt.legend(['REDOL', 'R. FOREST'])

    plt.title(f'Margin distribution for {sys.argv[1]} dataset')
    plt.xlabel(f'Margins')
    plt.ylabel(f'Distribution')

    # plt.show()
    plt.savefig(f'../plots/PNG/margin_distribution_{sys.argv[1]}_with_rf.png')
    plt.savefig(f'../plots/EPS/margin_distribution_{sys.argv[1]}_with_rf.eps')

    plt.close()

    plt.plot(np.sort(redol_margins_train), np.linspace(0, 1, len(redol_margins_train)))
    plt.plot(np.sort(margins_train), np.linspace(0, 1, len(redol_margins_train)))
    # plt.plot(np.sort(boostingmargins_train), np.linspace(0, 1, len(redol_margins_train)))
    # plt.plot(np.sort(baggingmargins_train), np.linspace(0, 1, len(redol_margins_train)))

    plt.legend(['REDOL', 'R. FOREST'])

    plt.title(f'Margin distribution for {sys.argv[1]} dataset in train')
    plt.xlabel(f'Margins')
    plt.ylabel(f'Distribution')

    # plt.show()
    plt.savefig(f'../plots/PNG/margin_distribution_{sys.argv[1]}_with_rf_train.png')
    plt.savefig(f'../plots/EPS/margin_distribution_{sys.argv[1]}_with_rf_train.eps')


if __name__ == "__main__":
    main()
