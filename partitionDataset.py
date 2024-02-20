import matplotlib.pyplot as plt
import seaborn as sns
from pyFTS.fcm import fts as fcm_fts
from pyFTS.partitioners import Grid
from pyFTS.common import Util
from pyFTS.common import Membership as mf
from scipy.optimize import least_squares
from scipy.optimize import leastsq
import pandas as pd
from pyFTS.benchmarks import Measures



y= data

rmse = []
mape = []
u = []
train=data[:6400]
test=data[6400:]

model = FCM_FTS(partitioner=partitioner, order=2, num_fcms=2,
                  activation_function=softplus,
                  loss=func)
model.fit(train)