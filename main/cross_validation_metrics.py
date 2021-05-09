import sys
sys.path.append("/home/cromero/projects/redol_model/redol")

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from redol import RedolClassifier

def get_data():
    if len(sys.argv) > 1:
        try:
            dataset = pd.read_csv(f"../data/original/{sys.argv[1]}.csv")

            cat_columns = ['class']

            if sys.argv[1] == "tic-tac-toe":
                cat_columns = dataset.select_dtypes(['object']).columns

            dataset[cat_columns] = dataset[cat_columns].astype('category')
            dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes)
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

    n_trees = 100
    n_splits = 5

    redolclf = RedolClassifier(n_estimators=n_trees, bootstrap=.7, perc=0.75, n_jobs=8, max_depth=None)
    distributedredolclf = RedolClassifier(n_estimators=n_trees, method="distributed", bootstrap=1, perc=0.75, n_jobs=8, max_depth=None)
    rfclf = RandomForestClassifier(n_estimators=n_trees)
    boostingclf = GradientBoostingClassifier(n_estimators=n_trees)
    baggingclf = BaggingClassifier(n_estimators=n_trees)

    redol_acc = []
    distredol_acc = []
    rf_acc = []
    boosting_acc = []
    bagging_acc = []

    redol_auc = []
    distredol_auc = []
    rf_auc = []
    boosting_auc = []
    bagging_auc = []

    skf = StratifiedKFold(n_splits=n_splits)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        redolclf.fit(X_train, y_train)
        distributedredolclf.fit(X_train, y_train)
        rfclf.fit(X_train, y_train)
        boostingclf.fit(X_train, y_train)
        baggingclf.fit(X_train, y_train)

        redol_acc.append(metrics.accuracy_score(y_test, redolclf.predict(X_test)))
        distredol_acc.append(metrics.accuracy_score(y_test, distributedredolclf.predict(X_test)))
        rf_acc.append(metrics.accuracy_score(y_test, rfclf.predict(X_test)))
        boosting_acc.append(metrics.accuracy_score(y_test, boostingclf.predict(X_test)))
        bagging_acc.append(metrics.accuracy_score(y_test, baggingclf.predict(X_test)))

        redol_auc.append(metrics.roc_auc_score(y_test, redolclf.predict_proba(X_test)[:, 1]))
        distredol_auc.append(metrics.roc_auc_score(y_test, distributedredolclf.predict_proba(X_test)[:, 1]))
        rf_auc.append(metrics.roc_auc_score(y_test, rfclf.predict_proba(X_test)[:, 1]))
        boosting_auc.append(metrics.roc_auc_score(y_test, boostingclf.predict_proba(X_test)[:, 1]))
        bagging_auc.append(metrics.roc_auc_score(y_test, baggingclf.predict_proba(X_test)[:, 1]))


    print(f"REDOL ACC ERROR: {1 - np.mean(redol_acc)}")
    print(f"REDOL DIST ACC ERROR: {1 - np.mean(distredol_acc)}")
    print(f"RANDOM FOREST ACC ERROR: {1 - np.mean(rf_acc)}")
    print(f"GRADIEN BOOSTING ACC ERROR: {1 - np.mean(boosting_acc)}")
    print(f"BAGGING ACC ERROR: {1 - np.mean(bagging_acc)}")
    print()
    print(f"REDOL AUC ERROR: {1 - np.mean(redol_auc)}")
    print(f"REDOL DIST AUC ERROR: {1 - np.mean(distredol_auc)}")
    print(f"RANDOM FOREST AUC ERROR: {1 - np.mean(rf_auc)}")
    print(f"GRADIEN BOOSTING AUC ERROR: {1 - np.mean(boosting_auc)}")
    print(f"BAGGING AUC ERROR: {1 - np.mean(bagging_auc)}")

if __name__ == "__main__":
    main()