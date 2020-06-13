import sys

sys.path.append('/home/cromero/noise-based-classifier/')

import resources.properties as properties
from resources.PlotModel import *

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from src.ClasificadorRuido import *
from sklearn.ensemble import RandomForestClassifier

from src.Alfredo import *
from src.Randbow import *


def get_data():
    if len(sys.argv) > 1:
        try:
            dataset = pd.read_csv(properties.DATASET_DIRECTORY + sys.argv[1])
            cat_columns = dataset.select_dtypes(['object']).columns
            dataset[cat_columns] = dataset[cat_columns].astype('category')
            dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes)
            dataset = dataset.values
            X, y = dataset[:,:-1], dataset[:,-1]
        except IOError:
            print("File \"{}\" does not exist.".format(sys.argv[1]))
            return
    else:
        X, y = make_moons(n_samples=5000, shuffle=True, noise=0.5, random_state=42)

    return X, y


def main():
    X, y = get_data()

    # X_tr, X_te, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_tr, X_te, y_train, y_test = train_test_split(X, y, test_size=0.5)

    ###################################
    #####      NORMALIZATION      #####
    ###################################
    # sc = StandardScaler()
    # X_tr_std = sc.fit_transform(X_tr)
    # X_te_std = sc.transform(X_te)

    #########################################
    #####      DATA TRANSFORMATION      #####
    #########################################
    # pca = PCA(n_components=2)
    #
    # X_train = pca.fit_transform(X_tr_std)
    # X_test = pca.transform(X_te_std)

    #########################################
    #####      DATA CLASSIFICATION      #####
    #########################################
    X_train = X_tr
    X_test = X_te

    n_trees = 1

    randbow = Randbow(n_trees=n_trees, perc=0.75)
    clf = Alfredo(n_trees=n_trees, perc=0.75)
    rfclf = RandomForestClassifier(n_estimators=n_trees)
    boostingclf = GradientBoostingClassifier(n_estimators=n_trees)
    baggingclf = BaggingClassifier(n_estimators=n_trees)

    print("Entrenamiento Randbow\n")
    randbow.fit(X_train, y_train)
    randbow.predict(X_test)

    print("Entrenamiento Alfredo\n")
    clf.fit(X_train, y_train)
    clf.predict(X_test)

    print("Entrenamiento RF\n")
    rfclf.fit(X_train, y_train)
    rfclf.predict(X_test)

    print("Entrenamiento Boosting\n")
    boostingclf.fit(X_train, y_train)
    boostingclf.predict(X_test)

    print("Entrenamiento Bagging\n")
    baggingclf.fit(X_train, y_train)
    baggingclf.predict(X_test)

    perm = PermutationImportance(randbow).fit(X_test, y_test)
    perm = PermutationImportance(clf).fit(X_test, y_test)
    perm_rf = PermutationImportance(rfclf).fit(X_test, y_test)
    perm_boost = PermutationImportance(boostingclf).fit(X_test, y_test)
    perm_bagg = PermutationImportance(baggingclf).fit(X_test, y_test)

    # print(properties.COLOR_BLUE, " Noise based importances", properties.END_C, perm.feature_importances_)
    # print(properties.COLOR_BLUE, " Random forest importances", properties.END_C, perm_rf.feature_importances_)
    # print(properties.COLOR_BLUE, " Boosting importances", properties.END_C, perm_boost.feature_importances_)
    # print(properties.COLOR_BLUE, " Bagging importances", properties.END_C, perm_bagg.feature_importances_)
    print("----------------------------------------------")
    print("{} Randbow:{} {}".format(properties.COLOR_BLUE, properties.END_C, clf.score(X_test, y_test)))
    print("{} Alfredo:{} {}".format(properties.COLOR_BLUE, properties.END_C, clf.score(X_test, y_test)))
    print("{} Random forest score:{} {}".format(properties.COLOR_BLUE, properties.END_C, rfclf.score(X_test, y_test)))
    print("{} Boosting score:{} {}".format(properties.COLOR_BLUE, properties.END_C, boostingclf.score(X_test, y_test)))
    print("{} Bagging score:{} {}".format(properties.COLOR_BLUE, properties.END_C, baggingclf.score(X_test, y_test)))

    plt.subplot(2, 3, 1)
    plot_model(randbow, X_train, y_train, "Randbow")

    plt.plot()

    plt.subplot(2, 3, 2)
    plot_model(clf, X_train, y_train, "Alfredo")

    plt.plot()

    plt.subplot(2, 3, 3)
    plot_model(rfclf, X_train, y_train, "Random forest")

    plt.plot()

    plt.subplot(2, 3, 4)
    plot_model(boostingclf, X_train, y_train, "Boosting")

    plt.plot()

    plt.subplot(2, 3, 5)
    plot_model(baggingclf, X_train, y_train, "Bagging")

    plt.plot()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
