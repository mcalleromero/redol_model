import sys

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree

from tqdm import tqdm

def get_data(model):
        try:
            dataset = pd.read_csv(f'../data/original/{model}.csv')

            cat_columns = ['class']

            if model == "tic-tac-toe":
                cat_columns = dataset.select_dtypes(['object']).columns

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

    k_folds = 30

    print(f'Stratifying {model}')

    skf = StratifiedShuffleSplit(n_splits=k_folds, test_size=0.33)

    train = []
    test = []
    for train_index, test_index in skf.split(X, y):
        train.append(train_index)
        test.append(test_index)

    train_path = f'../data/stratified_index/{model}_30folds_33test_train'
    test_path = f'../data/stratified_index/{model}_30folds_33test_test'

    np.save(train_path, np.array(train))
    np.save(test_path, np.array(test))

    print(f'Train file saved in {train_path}')
    print(f'Test file saved in {test_path}')


if __name__ == "__main__":
    main()
