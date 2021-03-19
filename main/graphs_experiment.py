import sys
sys.path.append('/home/cromero/projects/redol_model/')

import util.properties as properties
import numpy as np
import matplotlib.pyplot as plt

from redol import RedolClassifier


def main():
    # model = 'zeros0.5_unos0.5_ringnorm_normal_20dimm'
    model = sys.argv[1]

    data = np.load('../data/scores/30folds_33test/' + model + "_data_redol.npy")
    rfscore = np.load('../data/scores/30folds_33test/' + model + "_data_random-forest.npy")
    treescore = np.load('../data/scores/30folds_33test/' + model + "_data_tree.npy")
    boostingscore = np.load('../data/scores/30folds_33test/' + model + "_data_boosting.npy")
    baggingscore = np.load('../data/scores/30folds_33test/' + model + "_data_bagging.npy")

    """
    ----------------------------------------
                   COMPARISON
    ----------------------------------------
    """
    list_of_pil = [0.01, 0.15, 0.25, 0.5, 0.75, 0.85, 0.99]

    # This result is the mean from the kfold executions (i.e. 30)
    # which is the first index. The second index are the pil values
    # and the last index means the tree, being 99 the max if the
    # modelo was trained with 100 trees.
    res = data.mean(axis=0)[:, 99]

    # This is the index from the min error in the experiment
    print(np.where(res == np.amin(res))[0])

    print("MODELO: " + model)
    print()
    print("\tREDOL 50: " + str(data.mean(axis=0)[np.where(np.array(list_of_pil) == 0.5)[0][0], 99]))
    print()
    print("\tTREE: " + str(treescore.mean()))
    print("\tREDOL 75: " + str(data.mean(axis=0)[np.where(np.array(list_of_pil) == 0.75)[0][0], 99]))
    print("\tRANDOM F.: " + str(rfscore.mean()))
    print("\tBOOSTING: " + str(boostingscore.mean()))
    print("\tBAGGING: " + str(baggingscore.mean()))
    print()
    print("\tTREE STD: " + str(treescore.std()))
    print("\tREDOL 75 STD: " + str(data.std(axis=0)[np.where(np.array(list_of_pil) == 0.75)[0][0], 99]))
    print("\tRANDOM F. STD: " + str(rfscore.std()))
    print("\tBOOSTING STD: " + str(boostingscore.std()))
    print("\tBAGGING STD: " + str(baggingscore.std()))

    # # -------------------------------------- #
    #
    # print(data.shape)
    #
    model_title = str.upper(model[0]) + model[1:]
    plt.figure(figsize=(10, 5))

    """
    ----------------------------------------
                   FIRST PLOT
    ----------------------------------------
    """

    plt.subplot(1, 2, 1)

    plt.title(model_title + " pil evolution")

    lst = [0, 4, 9, 49, 99]

    for i in lst:
        plt.plot(list_of_pil, data.mean(axis=0)[0:, i], linestyle='-')

    plt.axhline(y=rfscore.mean(), color='m', linestyle='-')
    plt.axhline(y=boostingscore.mean(), color='g', linestyle='-')
    plt.axhline(y=baggingscore.mean(), color='y', linestyle='-')

    legend = list(map(lambda x: f'#trees={str(x+1)}', [0, 4, 9, 49, 99]))
    legend.append('RF')
    legend.append('gB.')
    legend.append('Bagg.')
    
    plt.legend(legend, loc='upper right', ncol=2)

    plt.ylabel("Err")
    plt.xlabel("Data randomization")

    # plt.ylim(0.0, 0.3)

    plt.grid()

    """
    ----------------------------------------
                   SECOND PLOT
    ----------------------------------------
    """

    plt.subplot(1, 2, 2)

    plt.title(model_title + " n. trees - err.")

    # lst = [1, 4, 9, 49, 74]
    
    for i in range(data.mean(axis=0).shape[0]):
        plt.plot(range(1, 101, 1), data.mean(axis=0)[i, :], linestyle='-')

    plt.axhline(y=rfscore.mean(), color='m', linestyle='-')
    plt.axhline(y=boostingscore.mean(), color='g', linestyle='-')
    plt.axhline(y=baggingscore.mean(), color='y', linestyle='-')

    legend = list(map(lambda x: f'pil={str(int(x*100))}%', list_of_pil))
    legend.append('RF')
    legend.append('gB.')
    legend.append('Bagg.')

    plt.legend(legend, loc='upper right', ncol=2)

    plt.ylabel("Err")
    plt.xlabel("N. trees")

    plt.grid()

    plt.tight_layout()

    plt.show()
    # plt.savefig("../plots/PNG/2-plots_" + model + ".png")
    # plt.savefig("../plots/EPS/2-plots_" + model + ".eps")

    # plt.savefig("../plots/PNG/unbalanced_plots_" + model + ".png")
    # plt.savefig("../plots/EPS/unbalanced_plots_" + model + ".eps")

if __name__ == "__main__":
    main()