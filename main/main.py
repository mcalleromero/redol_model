import sys
sys.path.append('/home/mario.calle/master/redol_model/')
import time
import util.properties as properties
from sklearn.model_selection import train_test_split
from sklearn import metrics
from mlxtend.evaluate import bias_variance_decomp
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from redol.redol import *
# from util import generator


def get_data():
    if len(sys.argv) > 1:
        try:
            dataset = pd.read_csv(
                f'{properties.DATASET_DIRECTORY}original/{sys.argv[1]}.csv')

            cat_columns = ['class']

            if sys.argv[1] == "tic-tac-toe":
                cat_columns = dataset.select_dtypes(['object']).columns

            dataset[cat_columns] = dataset[cat_columns].astype('category')
            dataset[cat_columns] = dataset[cat_columns].apply(
                lambda x: x.cat.codes)
            # xdataset = pd.get_dummies(dataset, columns=cat_columns)
            dataset = dataset.values

            X, y = dataset[:, :-1], dataset[:, -1]

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

    redolclf = RedolClassifier(n_estimators=n_trees, perc=0.75, n_jobs=8)
    rfclf = RandomForestClassifier(n_estimators=n_trees)
    boostingclf = GradientBoostingClassifier(n_estimators=n_trees)
    baggingclf = BaggingClassifier(n_estimators=n_trees)

    starttime = time.time()

    print("Entrenamiento Redol\n")
    redolclf.fit(X_train, y_train)
    redol_preds = redolclf.predict_proba(X_test)

    print('That took {} seconds'.format(time.time() - starttime))

    starttime = time.time()

    print("Entrenamiento RF\n")
    rfclf.fit(X_train, y_train)
    rf_preds = rfclf.predict_proba(X_test)

    print('That took {} seconds'.format(time.time() - starttime))

    starttime = time.time()

    print("Entrenamiento Boosting\n")
    boostingclf.fit(X_train, y_train)
    boost_preds = boostingclf.predict_proba(X_test)

    print('That took {} seconds'.format(time.time() - starttime))

    starttime = time.time()

    print("Entrenamiento Bagging\n")
    baggingclf.fit(X_train, y_train)
    bagg_preds = baggingclf.predict_proba(X_test)

    print('That took {} seconds'.format(time.time() - starttime))

    # mse, bias, var = bias_variance_decomp(redolclf, X_train, y_train, X_test, y_test, loss="mse")
    # print("----------------------------------------------")
    # print("{} REDOL MSE:{} {}".format(properties.COLOR_BLUE, properties.END_C, mse))
    # print("{} REDOL SUM:{} {}".format(properties.COLOR_BLUE, properties.END_C, var + bias))
    # print("{} REDOL BIAS:{} {}".format(properties.COLOR_BLUE, properties.END_C, bias))
    # print("{} REDOL VARIANCE:{} {}".format(properties.COLOR_BLUE, properties.END_C, var))
    # mse, bias, var = bias_variance_decomp(rfclf, X_train, y_train, X_test, y_test, loss="mse")
    # print("----------------------------------------------")
    # print("{} RANDOM FOREST MSE:{} {}".format(properties.COLOR_BLUE, properties.END_C, mse))
    # print("{} RANDOM FOREST SUM:{} {}".format(properties.COLOR_BLUE, properties.END_C, var + bias))
    # print("{} RANDOM FOREST BIAS:{} {}".format(properties.COLOR_BLUE, properties.END_C, bias))
    # print("{} RANDOM FOREST VARIANCE:{} {}".format(properties.COLOR_BLUE, properties.END_C, var))
    # mse, bias, var = bias_variance_decomp(boostingclf, X_train, y_train, X_test, y_test, loss="mse")
    # print("----------------------------------------------")
    # print("{} BOOSTING MSE:{} {}".format(properties.COLOR_BLUE, properties.END_C, mse))
    # print("{} BOOSTING SUM:{} {}".format(properties.COLOR_BLUE, properties.END_C, var + bias))
    # print("{} BOOSTING BIAS:{} {}".format(properties.COLOR_BLUE, properties.END_C, bias))
    # print("{} BOOSTING VARIANCE:{} {}".format(properties.COLOR_BLUE, properties.END_C, var))
    # mse, bias, var = bias_variance_decomp(baggingclf, X_train, y_train, X_test, y_test, loss="mse")
    # print("----------------------------------------------")
    # print("{} BAGGING MSE:{} {}".format(properties.COLOR_BLUE, properties.END_C, mse))
    # print("{} BAGGING SUM:{} {}".format(properties.COLOR_BLUE, properties.END_C, var + bias))
    # print("{} BAGGING BIAS:{} {}".format(properties.COLOR_BLUE, properties.END_C, bias))
    # print("{} BAGGING VARIANCE:{} {}".format(properties.COLOR_BLUE, properties.END_C, var))
    print("----------------------------------------------")
    print("----------------------------------------------")
    print("{} Redol accuracy:{} {}".format(properties.COLOR_BLUE,
                                           properties.END_C, redolclf.score(X_test, y_test)))
    print("{} Random forest accuracy:{} {}".format(
        properties.COLOR_BLUE, properties.END_C, rfclf.score(X_test, y_test)))
    print("{} Boosting accuracy:{} {}".format(properties.COLOR_BLUE,
                                              properties.END_C, boostingclf.score(X_test, y_test)))
    print("{} Bagging accuracy:{} {}".format(properties.COLOR_BLUE,
                                             properties.END_C, baggingclf.score(X_test, y_test)))
    print("----------------------------------------------")
    print("{} Redol log loss:{} {}".format(properties.COLOR_BLUE,
                                           properties.END_C, metrics.log_loss(y_test, redol_preds[:, 1])))
    print("{} Random forest log loss:{} {}".format(properties.COLOR_BLUE,
                                                   properties.END_C, metrics.log_loss(y_test, rf_preds[:, 1])))
    print("{} Boosting log loss:{} {}".format(properties.COLOR_BLUE,
                                              properties.END_C, metrics.log_loss(y_test, boost_preds[:, 1])))
    print("{} Bagging log loss:{} {}".format(properties.COLOR_BLUE,
                                             properties.END_C, metrics.log_loss(y_test, bagg_preds[:, 1])))
    print("----------------------------------------------")
    print("{} Redol AUC:{} {}".format(properties.COLOR_BLUE,
                                      properties.END_C, metrics.roc_auc_score(y_test, redol_preds[:, 1])))
    print("{} Random forest AUC:{} {}".format(properties.COLOR_BLUE,
                                              properties.END_C, metrics.roc_auc_score(y_test, rf_preds[:, 1])))
    print("{} Boosting AUC:{} {}".format(properties.COLOR_BLUE,
                                         properties.END_C, metrics.roc_auc_score(y_test, boost_preds[:, 1])))
    print("{} Bagging AUC:{} {}".format(properties.COLOR_BLUE,
                                        properties.END_C, metrics.roc_auc_score(y_test, bagg_preds[:, 1])))


if __name__ == "__main__":
    main()
