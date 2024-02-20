import os
from PIL import Image
import numpy as np
import shutil
import sys

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
                imgDir = path + '/' + key + '/' + key2
                imgNewPath = imgDir + '/' + imgName
                if not(os.path.isdir(imgDir)):
                    os.makedirs(imgDir)
                imgSave = img_PIL.save(imgNewPath)
                if os.path.isfile(imgNewPath):
                    print(imgName + ' salva!')


dataSetDir =  sys.argv[1] + '/'
trainDirList =  ['trainDataCppRAW', 'trainDataCppGridPYFTS', 'trainDataPolar']
indexString = str(sys.argv[2])
index = int(indexString)

trainDir = trainDirList[index]

path_to_dataSets = '/home/farsilva/felipe/doutorado/regular/modelo/UCR/'
rootPartition = path_to_dataSets + dataSetDir + trainDir

partitionDict = {}
filesList = []
numClients = 3
partitionSize = {}

for base, dirs, files in os.walk(rootPartition):
    for Dirs in dirs:
        partitionDict[Dirs] = []
        #print(Dirs)

for key in partitionDict:
    for base, dirs, files in os.walk(rootPartition + '/' + key):
        for Files in files:
            completePath = rootPartition + '/' + key + '/' + Files
            partitionDict[key].append(completePath)
    partitionSize[key] = np.zeros(numClients, dtype = int)
    partitionSize[key] += (int)(len(partitionDict[key])/numClients)
    if len(partitionDict[key]) % numClients != 0:
        partitionSize[key][:len(partitionDict[key]) % numClients] += 1

clientDict = {}

s = np.zeros(len(partitionSize), dtype = int)
for i in range(numClients):
    auxList = []
    auxDict = {}
    si = 0
    for key in partitionSize:
        auxList.append((key, partitionSize[key][i]))
        auxDict[key] = partitionDict[key][s[si]:s[si]+partitionSize[key][i]]
        s[si] = s[si] + partitionSize[key][i]
        si += 1
    #auxDict.update(auxList)
    clientDict['client' + str(i)] = auxDict
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

        
            
