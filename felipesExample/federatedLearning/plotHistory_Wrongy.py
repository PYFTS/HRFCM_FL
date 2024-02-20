import pandas as pd
import plotext as plt
import getDataset

history = pd.read_csv('AccHistory.csv')
plt.plot(history['acc'])
plt.grid()
plt.ylabel("Accuracy (%)")
plt.xlabel("Round")
plt.title(getDataset.datasetDir[:-1] + " 3 clients with 3 clients per round")
plt.show()