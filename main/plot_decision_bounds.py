import sys
sys.path.append('/home/mario.calle/master/redol_model/')

import time

import util.properties as properties
from util.PlotModel import *

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

from redol.redol import *
from util import generator


def plot_each_subplot(dataset, plot_index, perc):
    data = pd.read_csv(dataset)

    dataset = data.values

    X, y = dataset[:,:-1], dataset[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    n_trees = 100

    redolclf = RedolClassifier(n_estimators=n_trees, perc=perc, n_jobs=8)

    starttime = time.time()

    print("Entrenamiento Redol\n")
    redolclf.fit(X_train, y_train)
    redolclf.predict(X_test)

    print('That took {} seconds'.format(time.time() - starttime))

    print("{} Error obtained:{} {}".format(properties.COLOR_BLUE, properties.END_C, (1 - redolclf.score(X_test, y_test))))

    plt.subplot(3, 3, plot_index)

    plot_model(redolclf, X_test, y_test, "RedolClassifier", number_of_classes=data['class'].nunique(), predict_proba=True)

    plt.plot()
    

def main():

    model = 'ringnorm_normal_2dimm'

    datasets = ([f'zeros0.1_unos0.9_{model}.csv',
                f'zeros0.2_unos0.8_{model}.csv',
                f'zeros0.3_unos0.7_{model}.csv',
                f'zeros0.4_unos0.6_{model}.csv',
                f'zeros0.5_unos0.5_{model}.csv',
                f'zeros0.6_unos0.4_{model}.csv',
                f'zeros0.7_unos0.3_{model}.csv',
                f'zeros0.8_unos0.2_{model}.csv',
                f'zeros0.9_unos0.1_{model}.csv',])

    for perc in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for plot_index, dataset in enumerate(datasets):
            plot_each_subplot(f'../data/modified/{dataset}', plot_index + 1, perc)
        plt.savefig(f"../plots/EPS/decision_bounds_{perc}_{model}.eps")
        plt.savefig(f"../plots/PNG/decision_bounds_{perc}_{model}.png")
        plt.clf()

if __name__ == "__main__":
    main()
