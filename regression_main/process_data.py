import sys
sys.path.append('/home/cromero/projects/redol_model/')

import time
import util.properties as properties

from sklearn.model_selection import train_test_split

import pandas as pd

# from redol import RedolRegressor


def get_data():
    if len(sys.argv) > 1:
        try:
            dataset = pd.read_csv(f'../data/raw/regression/{sys.argv[1]}', sep=";")
            return dataset
        except IOError:
            print("File \"{}\" does not exist.".format(sys.argv[1]))
            return
    else:
        raise ValueError("File not found")

# cat_columns = ['class']

# if sys.argv[1] == "tic-tac-toe":
#     cat_columns = dataset.select_dtypes(['object']).columns

# dataset[cat_columns] = dataset[cat_columns].astype('category')
# dataset[cat_columns] = dataset[cat_columns].apply(lambda x: x.cat.codes)
# # xdataset = pd.get_dummies(dataset, columns=cat_columns)
# dataset = dataset.values

# X, y = dataset[:,:-1], dataset[:,-1]


if __name__ == "__main__":
    dataset = get_data()

    #dataset = dataset.rename(columns={"absences": "class"})
    print(dataset.sample(5))
    dataset.to_csv(f'../data/regression/{sys.argv[1]}', index=False)