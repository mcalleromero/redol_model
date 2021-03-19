import sys
sys.path.append('/home/cromero/projects/redol_model/')

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np

from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from redol import RedolClassifier

from tqdm import tqdm

def get_data(model):
        try:
            dataset = pd.read_csv(f'../data/original/{model}.csv')

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
    k_folds = 30
    
    train = np.load(f'../data/stratified_index/{model}_30folds_33test_train.npy')
    test = np.load(f'../data/stratified_index/{model}_30folds_33test_test.npy')

    print(train.shape)

    rf_scores = []
    tree_scores = []
    bagg_scores = []
    boost_scores = []

    rf_aucs = []
    tree_aucs = []
    bagging_aucs = []
    boosting_aucs = []

    list_of_pil = [0.01, 0.15, 0.25, 0.5, 0.75, 0.85, 0.99]

    clf_scores = np.empty((k_folds, len(list_of_pil), n_trees))
    clf_aucs = np.empty((k_folds, len(list_of_pil)))

    for i in tqdm(range(k_folds)):
        train_index = train[i, :]
        test_index = test[i, :]

        ### Training data generation ###

        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        ### Classifiers training and classification ###

        rfclf = RandomForestClassifier(n_estimators=n_trees, n_jobs=8)
        tree_clf = tree.DecisionTreeClassifier()
        boosting = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(), n_estimators=n_trees)
        bagging = BaggingClassifier(n_estimators=n_trees, n_jobs=8)

        rfclf.fit(X_train, y_train)
        rf_auc_preds = rfclf.predict_proba(X_test)
        # Save auc for random forest
        rf_aucs.append(metrics.roc_auc_score(y_test, rf_auc_preds[:, 1]))
        # Save 1-acc for random forest
        rf_scores.append(1 - rfclf.score(X_test, y_test))

        tree_clf.fit(X_train, y_train)
        tree_auc_preds = tree_clf.predict_proba(X_test)
        # Save auc for decision tree
        tree_aucs.append(metrics.roc_auc_score(y_test, tree_auc_preds[:, 1]))
        # Save 1-acc for decision tree
        tree_scores.append(1 - tree_clf.score(X_test, y_test))

        boosting.fit(X_train, y_train)
        boosting_auc_preds = boosting.predict_proba(X_test)
        # Save auc for decision boosting
        boosting_aucs.append(metrics.roc_auc_score(y_test, boosting_auc_preds[:, 1]))
        # Save 1-acc for decision boosting
        boost_scores.append(1 - boosting.score(X_test, y_test))

        bagging.fit(X_train, y_train)
        bagging_auc_preds = bagging.predict_proba(X_test)
        # Save auc for decision bagging
        bagging_aucs.append(metrics.roc_auc_score(y_test, bagging_auc_preds[:, 1]))
        # Save 1-acc for decision bagging
        bagg_scores.append(1 - bagging.score(X_test, y_test))

        ### Redol training and classification

        for perci, perc in enumerate(list_of_pil):
            clf = RedolClassifier(n_estimators=n_trees, perc=perc, bootstrap=1.0, n_jobs=8)

            clf.fit(X_train, y_train)

            # We save the AUC for each pil and for each fold
            y_pred = clf.predict_proba(X_test)
            clf_aucs[i, perci] = metrics.roc_auc_score(y_test, y_pred[:, 1])

            clf.predict_proba_error(X_test)

            scores = []

            for n_tree in range(1, n_trees + 1):
                scores.append(1 - clf.score_error(X_test, y_test, n_classifiers=n_tree))

            clf_scores[i, perci] = np.array(scores)


    np.save(f'../data/scores/30folds_33test/{model}_data_random-forest', np.array(rf_scores))
    np.save(f'../data/scores/30folds_33test/{model}_data_tree', np.array(tree_scores))
    np.save(f'../data/scores/30folds_33test/{model}_data_boosting', np.array(boost_scores))
    np.save(f'../data/scores/30folds_33test/{model}_data_bagging', np.array(bagg_scores))
    np.save(f'../data/scores/30folds_33test/{model}_data_redol', clf_scores)

    np.save(f'../data/auc/30folds_33test/auc_{model}_data_random-forest', np.array(rf_aucs))
    np.save(f'../data/auc/30folds_33test/auc_{model}_data_tree', np.array(tree_aucs))
    np.save(f'../data/auc/30folds_33test/auc_{model}_data_boosting', np.array(boosting_aucs))
    np.save(f'../data/auc/30folds_33test/auc_{model}_data_bagging', np.array(bagging_aucs))
    np.save(f'../data/auc/30folds_33test/auc_{model}_data_redol', clf_aucs)


if __name__ == "__main__":
    main()
