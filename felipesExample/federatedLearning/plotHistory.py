import matplotlib.pyplot as plt
#import plotext as plt
import getDataset
import pandas as pd


def plot(history):
    print(f"{history.metrics_centralized = }")
    global_accuracy_centralised = history.metrics_centralized["accuracy"]
    round = [data[0] for data in global_accuracy_centralised]
    acc = [100.0 * data[1] for data in global_accuracy_centralised]
    histDict = {}
    histDict['round'] = round
    histDict['acc'] = acc
    plt.scatter(round, acc)
    #plt.plotsize(100, 30)
    plt.grid()
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Round")
    plt.title(getDataset.datasetDir[:-1] + " 3 clients with 3 clients per round")
    plt.show()
    
    #histDF = pd.DataFrame(histDict)
    # histDF.to_csv('AccHistory.csv', index = False)