import sys
sys.path.append('/home/mario.calle/master/redol_model/')

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np

from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from redol.redol import RedolClassifier

from tqdm import tqdm

def get_data(model):
        try:
            dataset = pd.read_csv(f'../data/modified/{model}.csv')

            # AUXILIAR


            # aux = dataset['class']
            # dataset.drop(labels=['class'], axis=1, inplace = True)
            # dataset.insert(5, 'class', aux)

            cat_columns = ['class']

            if model == "tic-tac-toe":
                cat_columns = dataset.select_dtypes(['object']).columns

            # print(dataset)

            # AUXILIAR

            dataset[cat_columns] = dataset[cat_columns].astype('category')
            dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes)
            dataset = dataset.values
            X, y = dataset[:, :-1], dataset[:, -1]

            return X, y
        except IOError:
            print("File \"{}\" does not exist.".format(sys.argv[1]))
            exit(0)

def main():

    model = sys.argv[1]

    X, y = get_data(model)

    n_trees = 100
    k_folds = 100
    
    train = np.load(f'../data/stratified_index/{model}_train.npy')
    test = np.load(f'../data/stratified_index/{model}_test.npy')

    rf_scores = []
    tree_scores = []
    bagg_scores = []
    boost_scores = []

    clf_scores = np.empty((k_folds, len(np.arange(0.01, 0.999, 0.01)), n_trees))

    for i in tqdm(range(k_folds)):
        train_index = train[i, :]
        test_index = test[i, :]

        ### Training data generation ###

        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        ### Classifiers training and classification ###

        rfclf = RandomForestClassifier(n_estimators=n_trees, n_jobs=8)
        tree_clf = tree.DecisionTreeClassifier()
        boosting = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=5), n_estimators=n_trees)
        bagging = BaggingClassifier(n_estimators=n_trees, n_jobs=8)

        rfclf.fit(X_train, y_train)
        rf_scores.append(1 - rfclf.score(X_test, y_test))

        tree_clf.fit(X_train, y_train)
        tree_scores.append(1 - tree_clf.score(X_test, y_test))

        boosting.fit(X_train, y_train)
        boost_scores.append(1 - boosting.score(X_test, y_test))

        bagging.fit(X_train, y_train)
        bagg_scores.append(1 - bagging.score(X_test, y_test))

        ### Redol training and classification

        for perci, perc in enumerate(np.arange(0.01, 0.999, 0.01)):
            clf = RedolClassifier(n_estimators=n_trees, perc=perc, bagg=True, n_jobs=8)

            clf.fit(X_train, y_train)
            clf.predict_proba_error(X_test)

            scores = []

            for n_tree in range(1, n_trees + 1):
                scores.append(1 - clf.score_error(X_test, y_test, n_classifiers=n_tree))

            clf_scores[i, perci] = np.array(scores)


    np.save(f'../data/scores/{model}_data_random-forest', np.array(rf_scores))
    np.save(f'../data/scores/{model}_data_tree', np.array(tree_scores))
    np.save(f'../data/scores/{model}_data_boosting', np.array(boost_scores))
    np.save(f'../data/scores/{model}_data_bagging', np.array(bagg_scores))
    np.save(f'../data/scores/{model}_data', clf_scores)


if __name__ == "__main__":
    main()
