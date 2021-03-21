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


def get_data():
    if len(sys.argv) > 1:
        try:
            dataset = pd.read_csv(f'../data/original/{sys.argv[1]}.csv')

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
    model = sys.argv[1]


    redol_acc = []
    rf_acc = []
    redol_auc = []
    rf_auc = []

    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits)

    starttime = time.time()

    i = 0

    for train_index, test_index in skf.split(X, y):
        i += 1
        print(f"ITERACION {i}/{n_splits}")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        opt_redol = BayesSearchCV(
            estimator=RedolClassifier(), 
            search_spaces=[({
                # 'n_estimators': Integer(10, 500),
                'method': Categorical(["regular", "distributed"]),
                'perc': Real(0.5, 0.9, prior='log-uniform'),
                'bootstrap': Real(0.5, 1.0, prior='log-uniform'),
                'max_depth': Integer(2, 15),
            }, 50), ({
                # 'n_estimators': Integer(10, 500),
                'method': Categorical(["regular", "distributed"]),
                'perc': Real(0.5, 0.9, prior='log-uniform'),
                'bootstrap': Real(0.5, 1.0, prior='log-uniform'),
            }, 50)],
            n_iter=100,
            n_jobs=8,
        )

        # opt_redol = BayesSearchCV(
        #     estimator=RedolClassifier(), 
        #     search_spaces={
        #         'perc': Real(0.5, 0.9, prior='log-uniform'),
        #         'bootstrap': Real(0.5, 1.0, prior='log-uniform'),
        #         'max_depth': Integer(2, 15),
        #     },
        #     n_iter=100,
        #     n_jobs=-1,
        #     refit=True,
        # )

        opt_random_forest = BayesSearchCV(
            estimator=RandomForestClassifier(), 
            search_spaces=[({
                # 'n_estimators': Integer(10, 500),
                # 'criterion': Categorical(['gini', 'entropy']),
                'max_depth': Integer(2, 15),
                'max_features': Categorical([None, 'sqrt']),
            }, 50), ({
                'max_depth': Integer(2, 15),
                'max_features': Categorical([None, 'sqrt']),
                'max_samples': Real(0.5, 0.9, prior='log-uniform'),
            }, 50)],
            n_iter=100,
            n_jobs=8,
            refit=True,
        )

        print("Entrenamiento Redol\n")
        opt_redol.fit(X_train, y_train)
        redol_y_pred_probas = opt_redol.predict_proba(X_test)
        redol_y_pred = opt_redol.predict(X_test)
        redol_acc.append(1 - metrics.accuracy_score(y_test, redol_y_pred))
        redol_auc.append(1 - metrics.roc_auc_score(y_test, redol_y_pred_probas[:, 1]))

        print("Entrenamiento Random Forest\n")
        opt_random_forest.fit(X_train, y_train)
        random_forest_y_pred_probas = opt_random_forest.predict_proba(X_test)
        random_forest_y_pred = opt_random_forest.predict(X_test)
        rf_acc.append(1 - metrics.accuracy_score(y_test, random_forest_y_pred))
        rf_auc.append(1 - metrics.roc_auc_score(y_test, random_forest_y_pred_probas[:, 1]))

        print("----------------------------------------------")
        print("{} Redol best params:{} {}".format(properties.COLOR_BLUE, properties.END_C, opt_redol.best_params_))
        print("{} Random forest best params:{} {}".format(properties.COLOR_BLUE, properties.END_C, opt_random_forest.best_params_))
        print("{} Redol err:{} {}".format(properties.COLOR_BLUE, properties.END_C, 1 - metrics.accuracy_score(y_test, redol_y_pred)))
        print("{} Random forest err:{} {}".format(properties.COLOR_BLUE, properties.END_C, 1 - metrics.accuracy_score(y_test, random_forest_y_pred)))

    print('That took {} seconds'.format(time.time() - starttime))
    print("----------------------------------------------")
    print("{} Redol err:{} {}".format(properties.COLOR_BLUE, properties.END_C, np.mean(redol_acc)))
    print("{} Random forest err:{} {}".format(properties.COLOR_BLUE, properties.END_C, np.mean(rf_acc)))
    print("----------------------------------------------")
    print("{} Redol auc:{} {}".format(properties.COLOR_BLUE, properties.END_C, np.mean(redol_auc)))
    print("{} Random forest auc:{} {}".format(properties.COLOR_BLUE, properties.END_C, np.mean(rf_auc)))

    np.save(f"../data/results/hyperparameter_redol_{model}_metrics", np.array([redol_acc, redol_auc]))
    np.save(f"../data/results/hyperparameter_rf_{model}_metrics", np.array([rf_acc, rf_auc]))

if __name__ == "__main__":
    main()