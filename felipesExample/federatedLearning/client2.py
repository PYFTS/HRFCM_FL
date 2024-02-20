#%%
import os

import flwr as fl
import tensorflow as tf
from model import makeModel
from layers import createCNNLayers
from ypstruct import structure
import PIL
from PIL import Image
import numpy as np

from tensorflow import keras
from tensorflow.keras.utils import img_to_array, load_img
from keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import sys

#%%
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

dataSetDir = 'BeetleFly/'

trainDirList =  ['trainDataCppRAW_FL', 'trainDataCppGridPYFTS', 'trainDataCppCMeansPYFTS', 'trainDataCppNFN',
                 'trainDataPolar', 'trainDataWFTS_PYFTSGrid', 'trainDataWFTS_PYFTSCMeans',
                 'trainDataWFTS_NFNGrid']
testDirList =  ['testDataCppRAW', 'testDataCppGridPYFTS', 'testDataCppCMeansPYFTS', 'testDataCppNFN',
                'testDataPolar', 'testDataWFTS_PYFTSGrid', 'testDataWFTS_PYFTSCMeans',
                'testDataWFTS_NFNGrid']

listIndex = 0

partType = 'Grid'
trainDir = trainDirList[listIndex]
testDir = testDirList[listIndex]

#trainDir = 'imgFromCSV/MetodoPWFTS_TRAIN/'
#testDir = 'imgFromCSV/MetodoPWFTS_TEST/'

#path_to_data = '/content/drive/My Drive/ProjetoFelipeFabricio/'
path_to_dataSets = '/Users/felipe/Documents/Doutorado/regular/modelo/UCR/'

TRAIN_DATA_DIR = path_to_dataSets + dataSetDir + trainDir
VALIDATION_DATA_DIR = path_to_dataSets + dataSetDir + testDir

#%%
val_FOLDER = VALIDATION_DATA_DIR

totalValFiles = 0
totalValClasses = 0

for base, dirs, files in os.walk(val_FOLDER):
    for Files in files:
        totalValFiles += 1
    for Dirs in dirs:
        totalValClasses += 1

#%%
ColorMode = structure()

#ColorMode.name = 'rgb'
#ColorMode.shape = 3

ColorMode.name = 'grayscale'
ColorMode.shape = 1

#%%
testFile = base + '/' + files[0]
img = load_img(testFile,
               color_mode = ColorMode.name)

#%%
batch_size = 32
img_height = min(256,img_to_array(img).shape[1])
img_width = min(256,img_to_array(img).shape[0])

validation_generator = tf.keras.utils.image_dataset_from_directory(
  VALIDATION_DATA_DIR,
  seed=None,
  #shuffle = False,
  color_mode=ColorMode.name,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#%%
class_names = validation_generator.class_names
num_classes = len(class_names)

print(class_names)

for image_batch, labels_batch in validation_generator:
  img_size = image_batch.shape[1:]
  print(image_batch.shape)
  print(labels_batch.shape)
  break

#%%
clients = []

clientsRootFolder = TRAIN_DATA_DIR

totalTrainFiles = 0
totalTrainClasses = 0

for base, dirs, files in os.walk(clientsRootFolder):
    #for Files in files:
    #    totalTrainFiles += 1
    for Dirs in dirs:
        if Dirs.find('client') != -1:
            clients.append(Dirs)
            totalTrainClasses += 1

AUTOTUNE = tf.data.AUTOTUNE

val_ds = validation_generator.cache().prefetch(buffer_size=AUTOTUNE)

print(clients)
cid = int(sys.argv[1])

#%%
trainloaders = []

for client in clients:
    TRAIN_FOLDER = clientsRootFolder + '/'+ client + '/'
    for base, dirs, files in os.walk(TRAIN_FOLDER):
        for Files in files:
            totalTrainFiles += 1
        for Dirs in dirs:
            totalTrainClasses += 1

    train_generator = tf.keras.utils.image_dataset_from_directory(
        TRAIN_FOLDER,
        seed=None,
        #shuffle = False,
        color_mode=ColorMode.name,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    train_ds = train_generator.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

    trainloaders.append(train_ds)

print(len(trainloaders))

#%%
normalization_layer = layers.Rescaling(1./255)

normalized_ds = trainloaders[cid].map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

#%% Evaluation Function
# LeNet

convLayersList = []
convLayersList.append([16, 3, 'same', 'relu', 1, True, 'max', None, False])
convLayersList.append([32, 3, 'same', 'relu', 1, True, 'max', None, False])
convLayersList.append([64, 3, 'same', 'relu', 1, True, 'max', None, False])

convLayers, denseLayers = createCNNLayers(convLayersList, num_classes)
learningRateList = [1e-8, 0.0001, 0.001, 0.01, 0.1]

initialModel = Sequential(layers.Rescaling(scale = 1./255, input_shape=img_size))
cnnModel = makeModel(initialModel, convLayers, denseLayers, learningRateList[2])
cnnModel.compile(optimizer='adam', #Adam(learning_rate=learningRate),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
model = cnnModel

#%%
# Define Flower client
class Client(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(trainloaders[cid], epochs=1, batch_size=32)
        return model.get_weights(), totalTrainFiles, {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(val_ds)
        return loss, totalValFiles, {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=Client())