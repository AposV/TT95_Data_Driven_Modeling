import numpy as np
from ML_Helpers.Analyses import *
import seaborn as sns
import pandas as pd


def runMultipleTimes(analysis, run_times, name, path):
    training_r2_scores = []
    testing_r2_scores = []

    for i in range(run_times):

        if i==1:
            savegraphs=True
        else:
            savegraphs=False

        tr_r2, te_r2 = analysis.run(save_graphs=savegraphs)
        training_r2_scores.append(tr_r2)
        testing_r2_scores.append(te_r2)
        analysis.clear_scores()

    training_r2_scores = np.array(training_r2_scores)
    testing_r2_scores = np.array(testing_r2_scores)

    train = np.transpose(training_r2_scores).tolist()
    test = np.transpose(testing_r2_scores).tolist()

    trainmeans = [np.median(m) for m in train]
    testmeans = [np.median(m) for m in test]
    print(train)

    fig, ax = plt.subplots()

    sns.boxplot(data=train, color='blue', ax=ax, showfliers=False)
    sns.boxplot(data=test, color='green', ax=ax, showfliers=False)
    ax.plot(trainmeans, color='blue', label="Training")
    ax.plot(testmeans, color='green', label="Testing")

    ax.set_xlabel("Lookback Days")
    ax.set_ylabel("R^2 Score")

    ax.set_title("Comparison of model performance by look-back days")

    ax.legend()
    fig.savefig(analysis.path+"/summary.png")
    #plt.show()


