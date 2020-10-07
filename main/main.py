import os
import shutil

import time

import properties as properties

from sklearn.model_selection import train_test_split

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

from redol.redol import RedolClassifier


def get_data(file):
    try:
        dataset = pd.read_csv(file)

        cat_columns = ['class']

        # if file == "tic-tac-toe":
        #     cat_columns = dataset.select_dtypes(['object']).columns

        dataset[cat_columns] = dataset[cat_columns].astype('category')
        dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes)
        dataset = dataset.values

        X, y = dataset[:,:-1], dataset[:,-1]

        return X, y
    except IOError:
        print("File \"{}\" does not exist.".format(file))
        return

def main():

    base_path = os.getenv('INPUT_DATA')

    if not os.path.exists(base_path):
        try:
            os.mkdir(base_path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s " % path)

    if not os.path.exists(f'processed_data'):
        try:
            os.mkdir(f'processed_data')
        except OSError:
            print("Creation of the directory processed_data failed")
        else:
            print("Successfully created the directory processed_data")

    while True:
        if not os.listdir(base_path):
            print('There are no files to read. Sleep 5 seconds.')
            time.sleep(5)
            continue
        else:
            # Getting first file from the input_data folder
            filename = os.listdir(base_path)[0]
            f = f'{base_path}/{filename}'

            X, y = get_data(f)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            n_trees = 100

            redolclf = RedolClassifier(n_estimators=n_trees, perc=0.75, n_jobs=8)

            starttime = time.time()

            print("Fitting Redol\n")
            redolclf.fit(X_train, y_train)
            redolclf.predict(X_test)

            print('That took {} seconds'.format(time.time() - starttime))

            print("----------------------------------------------")
            print("{} Redol accuracy:{} {}".format(properties.COLOR_BLUE, properties.END_C, redolclf.score(X_test, y_test)))

            shutil.move(f, f'processed_data/{filename}')


if __name__ == "__main__":
    main()
