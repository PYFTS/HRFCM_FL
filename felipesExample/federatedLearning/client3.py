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
import getDataset

#%%
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

cid = int(sys.argv[1])
trainloaders = getDataset.getTrainLoaders()[0]

#%%
normalization_layer = layers.Rescaling(1./255)
normalized_ds = trainloaders[cid].map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
numEpochs = 50
VERBOSE = 0
# Notice the pixel values are now in `[0,1]`.
#print(np.min(first_image), np.max(first_image))

#%% Evaluation Function
# LeNet

convLayersList = []
convLayersList.append([16, 3, 'same', 'relu', 1, True, 'max', None, False])
convLayersList.append([32, 3, 'same', 'relu', 1, True, 'max', None, False])
convLayersList.append([64, 3, 'same', 'relu', 1, True, 'max', None, False])

convLayers, denseLayers = createCNNLayers(convLayersList, getDataset.getTrainLoaders()[2])
learningRateList = [1e-8, 0.0001, 0.001, 0.01, 0.1]

initialModel = Sequential(layers.Rescaling(scale = 1./255, input_shape=getDataset.getTrainLoaders()[1]))
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
        model.fit(trainloaders[cid], epochs=numEpochs, batch_size=32, verbose=VERBOSE)
        return model.get_weights(), getDataset.getTrainLoaders()[3], {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(getDataset.getTestLoader()[0])
        return loss, getDataset.getTestLoader()[3], {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=Client())