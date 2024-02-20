import os
from PIL import Image
import numpy as np
import shutil
import sys
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from fedlab.utils.dataset.partition import BasicPartitioner, CIFAR10Partitioner, CIFAR100Partitioner
from fedlab.utils.dataset import functional as F
from fedlab.utils.functional import partition_report

def deleteFolder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print('Folder and its content removed')
    except:
        print('Folder not deleted')

def saveImages(clientDict, path):
    for key in clientDict:
        for key2 in clientDict[key]:
            for imgFile in clientDict[key][key2]:
                img_PIL = Image.open(imgFile)
                imgName = imgFile.split('/')[-1]
                imgDir = path + '/' + str(key) + '/' + str(key2)
                imgNewPath = imgDir + '/' + imgName
                if not(os.path.isdir(imgDir)):
                    os.makedirs(imgDir)
                imgSave = img_PIL.save(imgNewPath)
                if os.path.isfile(imgNewPath):
                    print(imgName + ' salva!')

dataSetDir = sys.argv[1] + '/'
    
trainDirList =  ['trainDataCppRAW', 'trainDataCppGridPYFTS', 'trainDataPolar']
indexString = str(sys.argv[2])
    
index = int(indexString)
trainDir = trainDirList[index]

path_to_dataSets = '/home/felipe/felipe/doutorado/regular/modelo/UCR/'
rootPartition = path_to_dataSets + dataSetDir + trainDir

partitionDict = {}
filesList = []
numClients = 3
partitionSize = {}
targets = []
seed = 1

for base, dirs, files in os.walk(rootPartition):
    for Dirs in dirs:
        partitionDict[Dirs] = []
        #print(Dirs)


for key in partitionDict:
    for base, dirs, files in os.walk(rootPartition + '/' + key):
        for Files in files:
            completePath = rootPartition + '/' + key + '/' + Files
            partitionDict[key].append(completePath)
            targets.append(int(key))
            filesList.append(completePath)

targetsDF = pd.DataFrame(list(zip(targets, filesList)), columns =['Targets', 'Files'])
targetsDF = targetsDF.sort_values('Targets')
targetsOriginal = targetsDF['Targets'].to_list()
print(targetsOriginal)

le = LabelEncoder()
le.fit(targetsOriginal)
#targets = np.array(targetsDF['Targets'].to_list()) - np.min(np.array(targetsDF['Targets'].to_list()))
targets = le.transform(targetsOriginal)
print(targets)
filesList = targetsDF['Files'].to_list()
numClasses = len(partitionDict)
num_display_classes = min(numClasses, 10)
col_names = [f"class{i}" for i in range(numClasses)]
display_col_names = [f"class{i}" for i in range(num_display_classes)]

if numClasses == 2:
    hetero_dir_part = BasicPartitioner(targets=targets, num_clients=numClients, partition='noniid-labeldir', dir_alpha=0.3, seed=seed)
elif numClasses <= 10:
    hetero_dir_part = CIFAR10Partitioner(targets=targets, num_clients=numClients, partition='dirichlet', dir_alpha=0.3, seed=seed)
else:
    hetero_dir_part = CIFAR100Partitioner(targets=targets, num_clients=numClients, partition='dirichlet', dir_alpha=0.3, seed=seed)
    
print(hetero_dir_part.client_dict)

fileName = "BasePartition_dir_0.3_3clients"
csvFile = fileName + ".csv"
partition_report(targets, hetero_dir_part.client_dict, 
                 class_num=len(partitionDict), 
                 verbose=False, file=csvFile)

hetero_dir_part_df = pd.read_csv(csvFile,header=1)
hetero_dir_part_df = hetero_dir_part_df.set_index('client')
for col in col_names:
    hetero_dir_part_df[col] = (hetero_dir_part_df[col] * hetero_dir_part_df['Amount']).astype(int)

# select first 10 clients and first 10 classes for bar plot
hetero_dir_part_df[display_col_names].iloc[:3].plot.barh(stacked=True)  
# plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('sample num')
#plt.show()
plt.savefig(fileName + f".png", dpi=400, bbox_inches = 'tight')

clientDict = {}

s = np.zeros(len(partitionSize), dtype = int)
for key in hetero_dir_part.client_dict:
    auxList = []
    auxDict = {}
    for value in hetero_dir_part.client_dict[key]:
        auxList.append((targetsOriginal[value], filesList[value]))
        auxDict[targetsOriginal[value]] = []
    for pair in auxList:
        auxDict[pair[0]].append(pair[1])
    clientDict['client' + str(key)] = auxDict
    #for key in partitionSize:
    #    auxList.append((key, partitionSize[key][i]))
    #    auxDict[key] = partitionDict[key][s[si]:s[si]+partitionSize[key][i]]
    #auxDict.update(auxList)
    #clientDict['client' + str(i)] = auxDict
    
print('Número de classes: ' + str(numClasses))

for key in clientDict:
    for key2 in clientDict[key]:
        #for value in clientDict[key][key2]:
        print(key + ': ' + str(key2) + ': ' + str(len(clientDict[key][key2])))
    
print(clientDict)
newPath = path_to_dataSets + dataSetDir + trainDir + '_FL'
if os.path.isdir(newPath):
    delete = input('Continuar a criar diretórios e deletar antigos? ')
    if delete == 'S' or delete == 's':
        deleteFolder(newPath)
        saveImages(clientDict, newPath)
    else:
        print('Diretórios não excluídos')
else:
    saveImages(clientDict, newPath)

        
            
