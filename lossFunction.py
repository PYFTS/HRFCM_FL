import numpy as np

def func(values, *arg):
    #print("Values:")
    #print(values.shape)
    #print("arg:")
    #print(len(arg))
    if len(values.shape) == 1:
      n_coefs = values.shape[0] + 1
      #return arg[0] + np.sum([values[i-1] * arg[i] for i in np.arange(1,n_coefs)])
      ret = 0
    else:
      n_coefs = values.shape[0] + 1
      n_inst = values.shape[1]
      ret = np.zeros(n_inst)
    for k in range(1, n_coefs):
      ret = ret + arg[k] * values[k-1]

    ret = ret + arg[0]
    
    return ret
  