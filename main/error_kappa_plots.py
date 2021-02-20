import sys
sys.path.append('/home/mario.calle/master/redol_model/')

import time

import util.properties as properties
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics
from mlxtend.evaluate import bias_variance_decomp

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

from redol.redol import *
from util import generator

np.random.seed = 1


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

def contingency_table(y1, y2, y_real):
    cont = []

    a = np.where((y1 == y_real) & (y2 == y_real))[0].shape[0]
    b = np.where((y1 == y_real) & (y2 != y_real))[0].shape[0]
    c = np.where((y1 != y_real) & (y2 == y_real))[0].shape[0]
    d = np.where((y1 != y_real) & (y2 != y_real))[0].shape[0]

    N = y_real.shape[0]

    # e = 0.5 * (((c + d) / N) + ((b + d) / N))
    e = (b + c + 2*d) / (2*N)
    k = (2 * (a*d - b*c)) / (((a + b) * (b + d)) + ((a + c) * (c + d)))
    
    return np.array([a, b, c, d]).reshape(2, 2), e, k

def main():

    X, y = get_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    n_trees = 100

    redolclf = RedolClassifier(n_estimators=n_trees, perc=0.75, n_jobs=8)
    rfclf = RandomForestClassifier(n_estimators=n_trees)
    boostingclf = GradientBoostingClassifier(n_estimators=n_trees)
    baggingclf = BaggingClassifier(n_estimators=n_trees)

    redolclf.fit(X_train, y_train)
    rfclf.fit(X_train, y_train)
    # print(redolclf.predict_by_clf(X_test))
    # print(contingency_table(redolclf.predict_by_clf(X_test)[23,:], redolclf.predict_by_clf(X_test)[1,:],y_test))

    data = X_test
    data_test = y_test

    prev_preds = redolclf.predict(data)
    # predicts = redolclf.predict_by_clf(data, prev_preds=prev_preds)
    predicts = redolclf.predict_by_clf(data, prev_preds=y_test)
    
    rfpredicts = []
    for clf in rfclf.estimators_:
        rfpredicts.append(clf.predict(data))

    rfpredicts = np.array(rfpredicts)
    # 'tab:blue', 'tab:orange', 'tab:green'

    error = []
    k_coords = []
    error_rf = []
    k_coords_rf = []

    for i in range(100):
        for j in range(i+1, 100):
            _, e, k = contingency_table(predicts[i,:], predicts[j,:], data_test)
            _, e_rf, k_rf = contingency_table(rfpredicts[i,:], rfpredicts[j,:], data_test)

            error.append(e)
            k_coords.append(k)

            error_rf.append(e_rf)
            k_coords_rf.append(k_rf)

    print(len(error))

    plt.title(f'{sys.argv[1].capitalize()} error - kappa plot')
    plt.xlabel('k')
    plt.ylabel('error')
    plt.xlim(-1, 1)
    plt.ylim(0, 1)
    plt.plot(k_coords, error, 'o', c='tab:green', markersize=1, alpha=0.5)
    plt.plot(k_coords_rf, error_rf, 'o', c='tab:blue', markersize=1, alpha=0.5)
    plt.legend(['REDOL', 'Random Forest'])
    plt.savefig(f'../plots/PNG/kappa_{sys.argv[1]}_test.png')
    plt.savefig(f'../plots/EPS/kappa_{sys.argv[1]}_test.eps')

    # plt.savefig(f'../plots/PNG/kappa_{sys.argv[1]}_test.png')
    # plt.savefig(f'../plots/EPS/kappa_{sys.argv[1]}_test.eps')
# 
    # starttime = time.time()
# 
    # print("Entrenamiento Redol\n")
    # redol_preds = redolclf.predict_proba(X_test)
# 
    # print('That took {} seconds'.format(time.time() - starttime))
# 
    # starttime = time.time()
# 
    # print("Entrenamiento RF\n")
    # rfclf.fit(X_train, y_train)
    # rf_preds = rfclf.predict_proba(X_test)
# 
    # print('That took {} seconds'.format(time.time() - starttime))
# 
    # starttime = time.time()
# 
    # print("Entrenamiento Boosting\n")
    # boostingclf.fit(X_train, y_train)
    # boost_preds = boostingclf.predict_proba(X_test)
# 
    # print('That took {} seconds'.format(time.time() - starttime))
# 
    # starttime = time.time()
# 
    # print("Entrenamiento Bagging\n")
    # baggingclf.fit(X_train, y_train)
    # bagg_preds = baggingclf.predict_proba(X_test)
# 
    # print('That took {} seconds'.format(time.time() - starttime))


if __name__ == "__main__":
    main()
