import sys
sys.path.append('/home/cromero/projects/redol_model/')

import time
import util.properties as properties

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

from redol import RedolClassifier


def main():

    models = [
        "wine", 
        "new-thyroid", 
        "heart",
        "ionosphere",
        "australian",
        "wdbc",
        "diabetes",
        "tic-tac-toe",
        "german",
        "segment",
        "magic04",
        "twonorm",
        "threenorm",
        "ringnorm",
        "waveform",
        ]

    print("MODEL\t\t\tREDOL\t\tRANDOM F.")
    for model in models:
        redol = np.load(f"../data/results/hyperparameter_redol_{model}_metrics3.npy")
        rf = np.load(f"../data/results/hyperparameter_rf_{model}_metrics3.npy")

        print(f"{model}\t\t\t{round(np.mean(redol)*100, 3)}+-{round(np.std(redol)*100, 3)}\t\t{round(np.mean(rf)*100, 3)}+-{round(np.std(rf)*100, 3)}")

if __name__ == "__main__":
    main()
