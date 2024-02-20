import lossFunction
from pyFTS.models import hofts
from FCM import FuzzyCognitiveMap
import numpy as np
from numpy.linalg import svd

class FCM_FTS(hofts.HighOrderFTS):

    def __init__(self, **kwargs):
        super(FCM_FTS, self).__init__(**kwargs)

        num_concepts = self.partitioner.partitions

        self.num_fcms = kwargs.get('num_fcms', 2)

        self.fcm = []

        self.loss_function = kwargs.get('loss', lossFunction.func)

        for k in range(self.num_fcms):
          fcm_tmp = FuzzyCognitiveMap(**kwargs)
          weights = np.random.uniform(-1, 1, size=(self.order,num_concepts, num_concepts))
          specturalradius1=np.max(np.abs(np.linalg.eigvals(weights)))
          fcm_tmp.weights = weights*0.5/specturalradius1
          bias = np.random.uniform(-1, 1, size=(self.order,num_concepts))
          U,S,VT=svd(bias)
          specturalradius2=np.max(S)
          fcm_tmp.bias=bias*0.5/specturalradius2
          self.fcm.append(fcm_tmp)


        # Coefficients
        #self.theta = np.zeros(self.num_fcms + 1)
        self.theta = np.random.rand(self.num_fcms + 1)

    def forecast(self, data, **kwargs):
        y1 = []

        midpoints = np.array([fset.centroid for fset in self.partitioner])

        for t in np.arange(self.order, len(data)+1):

            sample = data[t - self.order : t]

            fuzzyfied = self.partitioner.fuzzyfy(sample, mode='vector')

            # Evaluate FCMs

            forecasts = []

            for fcm in self.fcm:
              activation=fcm.activate(fuzzyfied)
              forecasts.append(np.dot(midpoints, activation)/np.nanmax([1, np.sum(activation)]))

            # Combine the results

            #print(forecasts)

            result = self.loss_function(np.array(forecasts), *self.theta)

            if str(result) == 'nan' or result == np.nan or result == np.Inf:
               print('error')

            y1.append(result)

        return y1

    def run_fcm(self, fcm, data):
        ret = []
        midpoints = np.array([fset.centroid for fset in self.partitioner])
        for t in np.arange(self.order, len(data)+1):
            sample = data[t - self.order : t]
            fuzzyfied = self.partitioner.fuzzyfy(sample, mode='vector')
            activation = fcm.activate(fuzzyfied)
            final = np.dot(midpoints, activation)/np.nanmax([1, np.sum(activation)])
            ret.append(final)
        return ret

    def train(self, data, **kwargs):
        from scipy.optimize import curve_fit, least_squares, minimize, leastsq

        outputs = []

        for model in self.fcm:
          outputs.append(self.run_fcm(model, data)[:-1])

        #print("Outputs:")
        #print(len(outputs[1]))
        #print("coef")
        #print(self.theta)
        
        f = lambda coef, y, x: self.loss_function(x, *coef) - y
        
        self.theta, flag = leastsq(f, x0 = self.theta, args=(data[self.order:], np.array(outputs)))
        
        
        # print(self.theta) # least squares coefficients

    def get_parameters(self):
        parameters = []
        for l,fcm in enumerate(self.fcm):
            #parameters['weights'] = fcm.weights
            parameters.append(fcm.weights)
            #parameters['bias'] = fcm.bias
            parameters.append(fcm.bias)
        #parameters['theta'] = self.theta
        parameters.append(self.theta)
        #print('Get Parameters:')
        #print(parameters)
        return parameters
    
    def set_parameters(self, parameters):
        #print('Set Parameters:')
        #print(parameters)
        for l,fcm in enumerate(self.fcm):
            fcm.weights = parameters[l*2]
            fcm.bias = parameters[l*2+1]
        self.theta = parameters[-1]

    #def evaluate(self, test):