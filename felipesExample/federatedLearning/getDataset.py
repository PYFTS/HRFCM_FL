import os
from ypstruct import structure
import tensorflow as tf
from tensorflow.keras.utils import img_to_array, load_img

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32

path_to_dataSets = '/home/felipe/felipe/doutorado/regular/modelo/UCR/'
datasetDir = 'ArrowHead/'
listIndex = 0

ColorMode = structure()

#ColorMode.name = 'rgb'
#ColorMode.shape = 3

ColorMode.name = 'grayscale'
ColorMode.shape = 1

def getTrainLoaders():

    trainDirList =  ['trainDataCppRAW_FL', 'trainDataCppGridPYFTS_FL', 'trainDataPolar_FL']

    trainDir = trainDirList[listIndex]

    TRAIN_DATA_DIR = path_to_dataSets + datasetDir + trainDir

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

    print(clients)

    trainloaders = []

    for client in clients:
        TRAIN_FOLDER = clientsRootFolder + '/' + client + '/'
        for base, dirs, files in os.walk(TRAIN_FOLDER):
            for Files in files:
                totalTrainFiles += 1
            for Dirs in dirs:
                totalTrainClasses += 1
    
        testFile = base + '/' + files[0]
        img = load_img(testFile,
                   color_mode = ColorMode.name)

        img_height = min(256,img_to_array(img).shape[1])
        img_width = min(256,img_to_array(img).shape[0]) 

        train_generator = tf.keras.utils.image_dataset_from_directory(
            TRAIN_FOLDER,
            seed=None,
            #shuffle = False,
            color_mode=ColorMode.name,
            image_size=(img_height, img_width),
            batch_size=batch_size)
    
        for image_batch, labels_batch in train_generator:
            img_size = image_batch.shape[1:]
            break

        class_names = train_generator.class_names
        num_classes = len(class_names)

        train_ds = train_generator.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

        trainloaders.append(train_ds)

    return trainloaders, img_size, num_classes, totalTrainFiles

def getTestLoader():

    testDirList =  ['testDataCppRAW', 'testDataCppGridPYFTS', 
                    'testDataPolar']


    testDir = testDirList[listIndex]

    TEST_DATA_DIR = path_to_dataSets + datasetDir + testDir

    test_FOLDER = TEST_DATA_DIR

    totalTestFiles = 0
    totalTestClasses = 0

    for base, dirs, files in os.walk(test_FOLDER): 
        for Files in files:
            totalTestFiles += 1
        for Dirs in dirs:
            totalTestClasses += 1

    testFile = base + '/' + files[0]
    img = load_img(testFile,
                   color_mode = ColorMode.name)
    
    img_height = min(256,img_to_array(img).shape[1])
    img_width = min(256,img_to_array(img).shape[0])

    test_generator = tf.keras.utils.image_dataset_from_directory(
        TEST_DATA_DIR,
        seed=None,
        #shuffle = False,
        color_mode=ColorMode.name,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = test_generator.class_names
    num_classes = len(class_names)

    #print(class_names)

    for image_batch, labels_batch in test_generator:
        img_size = image_batch.shape[1:]
        #print(image_batch.shape)
        #print(labels_batch.shape)
        break

    test_ds = test_generator.cache().prefetch(buffer_size=AUTOTUNE)
    
    return test_ds, img_size, num_classes, totalTestFiles


#normalization_layer = layers.Rescaling(1./255)
#test = getTrainLoaders()
#print(len(test[0]))
#print(test[0][0].map(lambda x, y: (normalization_layer(x), y)))





