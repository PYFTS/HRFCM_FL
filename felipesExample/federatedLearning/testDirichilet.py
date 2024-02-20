import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import sys


from fedlab.utils.dataset.partition import CIFAR100Partitioner
from fedlab.utils.dataset import functional as F
from fedlab.utils.functional import partition_report

trainset = torchvision.datasets.CIFAR100(root="../../../../data/CIFAR100/", train=True, download=True)

#print(len(trainset.targets))

num_clients = 100
num_classes = 100
num_display_classes = 10


col_names = [f"class{i}" for i in range(num_classes)]
display_col_names = [f"class{i}" for i in range(num_display_classes)]

seed = 2021
print(trainset.targets)
hist_color = '#4169E1'
#plt.rcParams['figure.facecolor'] = 'white'

# perform partition
hetero_dir_part = CIFAR100Partitioner(trainset.targets, 
                                num_clients,
                                balance=None, 
                                partition="dirichlet",
                                dir_alpha=0.3,
                                seed=seed)
# save to pkl file
torch.save(hetero_dir_part.client_dict, "cifar100_hetero_dir.pkl")
print(len(hetero_dir_part))
#print(hetero_dir_part.client_dict)

csv_file = "cifar100_hetero_dir_0.3_100clients.csv"
partition_report(trainset.targets, hetero_dir_part.client_dict, 
                 class_num=num_classes, 
                 verbose=False, file=csv_file)

hetero_dir_part_df = pd.read_csv(csv_file,header=1)
hetero_dir_part_df = hetero_dir_part_df.set_index('client')
for col in col_names:
    hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['Amount']).astype(int)

#print(hetero_dir_part_df)
#s = np.random.dirichlet((10, 10), 20).transpose()

#plt.barh(range(20), s[0])
#plt.barh(range(20), s[1], left=s[0], color='g')
#plt.barh(range(20), s[2], left=s[0]+s[1], color='r')
#plt.title("Lengths of Strings")
#print(np.sum(s, axis = 1))
#plt.show()
