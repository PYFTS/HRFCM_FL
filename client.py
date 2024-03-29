import os

import flwr as fl
import numpy as np

from pyFTS.fcm import fts as fcm_fts
from pyFTS.partitioners import Grid
from pyFTS.common import Util
from pyFTS.common import Membership as mf
from scipy.optimize import least_squares
from scipy.optimize import leastsq
import pandas as pd
from pyFTS.benchmarks import Measures
from pyFTS.fcm import Activations

import sys

import FCM_FTS, FCM
import lossFunction

#%%
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

cid = int(sys.argv[1])

# Create clients partition
df1 = pd.read_csv('nrel_DHHL_1.csv')
df2 = pd.read_csv('nrel_DHHL_2.csv')
df3 = pd.read_csv('nrel_DHHL_3.csv')
#df4 = pd.read_csv('https://query.data.world/s/56i2vkijbvxhtv5gagn7ggk3zw3ksi', sep=';')

clients = {}
clients[0] = df1['value'].values[:8000]
clients[1] = df2['value'].values[:8000]
clients[2] = df3['value'].values[:8000]
#clients[3] = df4['glo_avg'].values[:8000]

partitioner = Grid.GridPartitioner(data=clients[cid], npart=3, mf=mf.trimf)

train = clients[cid][:6400]
test = clients[cid][6400:]

model = FCM_FTS.FCM_FTS(partitioner=partitioner, order=3, num_fcms=20,
                  activation_function=Activations.relu,
                  loss=lossFunction.func)

#parameters = model.get_parameters()
#%%
# Define Flower client
class Client(fl.client.NumPyClient):
    #def __init__(self, parameters):
    #    self.parameters = parameters

    def get_parameters(self, config):
        return model.get_parameters()

    def fit(self, parameters, config):
        #print("Client: ")
        #print(parameters)
        #print("\n")
        model.set_parameters(parameters)
        model.fit(clients[cid])
        return model.get_parameters(), len(clients[cid]), {}

    def evaluate(self, parameters, config):
        model.set_parameters(parameters)
        #print(model.get_parameters())
        forecasted = model.predict(test)
        _rmse  = Measures.rmse(test, forecasted, model.order-1)
        #rmse = model.evaluate(test)
        return _rmse, len(test), {"rmse": _rmse}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=Client())