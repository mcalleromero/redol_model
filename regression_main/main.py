import sys
sys.path.append('/home/cromero/projects/redol_model/')

import time
import util.properties as properties

from sklearn.model_selection import train_test_split
from sklearn import metrics

import pandas as pd

from redol import RedolRegressor
from sklearn.tree import DecisionTreeRegressor

def get_data():
    if len(sys.argv) > 1:
        try:
            dataset = pd.read_csv(f'../data/regression/{sys.argv[1]}.csv')

            X = dataset.drop(columns="class")
            y = dataset["class"]

            return X.to_numpy(), y.to_numpy()
        except IOError:
            print("File \"{}\" does not exist.".format(sys.argv[1]))
            return
    else:
        raise ValueError("File not found")

if __name__=="__main__":
    X, y = get_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    n_trees = 100

    redol = RedolRegressor(n_estimators=n_trees, perc=0.75, repetitions=3, nearest_neighbours=5, iterations=10, n_jobs=4)
    redol.fit(X_train, y_train)
    y_pred = redol.predict(X_test)
    print(metrics.mean_squared_error(y_test, y_pred))

    tree = DecisionTreeRegressor()
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print(metrics.mean_squared_error(y_test, y_pred))
