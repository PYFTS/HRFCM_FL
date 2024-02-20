import matplotlib.pyplot as plt
#import seaborn as sns
from pyFTS.fcm import fts as fcm_fts
from pyFTS.partitioners import Grid
from pyFTS.common import Util
from pyFTS.common import Membership as mf
from scipy.optimize import least_squares
from scipy.optimize import leastsq
import pandas as pd
from pyFTS.benchmarks import Measures
from pyFTS.fcm import Activations

import lossFunction
import FCM_FTS

df = pd.read_csv('https://query.data.world/s/56i2vkijbvxhtv5gagn7ggk3zw3ksi', sep=';')

data = df['glo_avg'].values[:8000]
partitioner = Grid.GridPartitioner(data=data, npart=3, mf=mf.trimf)
y= data

rmse = []
mape = []
u = []
train=data[:6400]
test=data[6400:]

model = FCM_FTS.FCM_FTS(partitioner=partitioner, order=2, num_fcms=2,
                  activation_function=Activations.sigmoid,
                  loss=lossFunction.func)
model.fit(train)