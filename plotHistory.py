import matplotlib.pyplot as plt
#import plotext as plt
import pandas as pd

def saveHistory(historyMetric, metricType, i):
    #print(historyMetric)
    round = [data[0] for data in historyMetric['rmse']]
    rmse = [data[1] for data in historyMetric['rmse']]
    nrmse = [data[1] for data in historyMetric['nrmse']]
    clientRMSE = [data[1] for data in historyMetric['clientsRMSE']]
    clientNRMSE = [data[1] for data in historyMetric['clientsNRMSE']]
    clientId = [data[1] for data in historyMetric['ClientId']]
    histDict = {}
    histDict['round'] = round
    histDict['rmse'] = rmse
    histDict['nrmse'] = nrmse
    histDict['ClientId'] = clientId
    histDict['Clients rmse'] = clientRMSE
    histDict['Clients nrmse'] = clientNRMSE
    plt.scatter(round, nrmse)
    #plt.plotsize(100, 30)
    plt.grid()
    plt.ylabel("nrmse (%)")
    plt.xlabel("Round")
    plt.title("nrmse 3 clients with 3 clients per round")
    #plt.show()
    #print(histDict)
    histDF = pd.DataFrame(histDict)
    histDF.to_csv('ResulsHistory' + metricType + '_Exp' + str(i + 1) + '.csv', index = False)

def plot(history, expNumber):
    print('====================== Hist Metrics =========================')
    print(history.metrics_distributed)
    #global_accuracy_centralised = history.metrics_centralized["accuracy"]
    #saveHistory(global_accuracy_centralised, 'Centralized', expNumber)
    global_metrics_distributed = history.metrics_distributed #["rmse","nrmse"]
    saveHistory(global_metrics_distributed, 'Distributed', expNumber)
