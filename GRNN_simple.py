#plt.legend(['Predicted RSSI'], loc='upper left')

# Model Imports
import tensorflow as tf
from keras import backend as K

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Activation, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.utils import multi_gpu_model
from keras.callbacks import LearningRateScheduler

from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np
import warnings
import h5py


# Data Preprocessing

#3D Floorplan Input Matrix
X_train = np.load("X_train.npy")
X_train = X_train[:-1]
#print("X train : ",X_train.shape)

X_test = np.load("X_test.npy")
#print("X test : ", X_test.shape)

Y_train = np.load("Y_train.npy")
Y_train = Y_train[:-1]
Y_train = Y_train.reshape(len(Y_train),1)
#print("Y train : ", Y_train.shape)

Y_test = np.load("Y_test.npy")
Y_test = Y_test.reshape(len(Y_test),1)
#print("Y test : ", Y_test.shape)

#scaler_y = MinMaxScaler()
#Y_train = scaler_y.fit_transform(Y_train)
#Y_test = scaler_y.transform(Y_test)

# Model Initialization.
batch_size = 4
no_epochs = 100
learning_rate = 0.001#0.0001
no_classes = 10
validation_split = 0.3
verbosity = 1
sample_shape = (26, 255, 255, 3)
steps_per_epoch = len(X_train)//batch_size
NN_model = Sequential()

# The Input Layer :
NN_model.add(Conv3D(12, kernel_size=(3, 3, 3), activation = 'relu', kernel_initializer = 'he_uniform', input_shape = sample_shape))
NN_model.add(BatchNormalization())
NN_model.add(MaxPooling3D(pool_size=(2, 2, 2)))

# The Hidden Layers :
NN_model.add(Conv3D(20, kernel_size=(3, 3, 3), activation = 'relu', kernel_initializer = 'he_uniform'))
NN_model.add(BatchNormalization())
NN_model.add(MaxPooling3D(pool_size=(2, 2, 2)))
#NN_model.add(Dropout(0.4))

#NN_model.add(Conv3D(16, kernel_size=(3, 3, 3), activation = 'relu', kernel_initializer = 'he_uniform'))
#NN_model.add(BatchNormalization())
#NN_model.add(MaxPooling3D(pool_size=(1, 1, 1)))
#NN_model.add(Dropout(0.4))

#NN_model.add(Conv3D(32, kernel_size=(3, 3, 3), activation = 'relu', kernel_initializer = 'he_uniform'))
#NN_model.add(BatchNormalization())
#NN_model.add(MaxPooling3D(pool_size=(1, 1, 1)))
#NN_model.add(Dropout(0.4))


NN_model.add(Flatten())

NN_model.add(Dense(20, kernel_initializer='he_uniform',activation='relu'))

#NN_model.add(Dense(32, kernel_initializer='he_uniform',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer = 'he_uniform', activation = 'linear'))

# Compile the network :
NN_model.compile(optimizer = 'adam',loss='mae')
NN_model.summary()

checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor = 'val_loss', verbose = 1, save_best_only = True, mode ='auto')
'''
def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 90
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr
'''
callbacks_list = [checkpoint]

def train(NN_model):
    

    # Replicates `model` on 4 GPUs.
    # This assumes that your machine has 4 available GPUs.
    NN_model = multi_gpu_model(NN_model, gpus=2)
    #parallel_model.compile(loss='mae',optimizer='adam')

    
    #Load wights file of the best model :
    #wights_file = 'Weights-002--8.27953.hdf5' # choose the best checkpoint
    #NN_model.load_weights(wights_file) # load it
    NN_model.compile(optimizer = 'adam',loss='mae')
                    
    # Train the model
    history = NN_model.fit(X_train,Y_train, epochs = no_epochs,batch_size=batch_size, validation_split = validation_split,verbose=1,callbacks = callbacks_list)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    
def test(NN_model):

    #Load wights file of the best model :
    wights_file = 'Weights-097--4.43342.hdf5' # choose the best checkpoint 
    NN_model.load_weights(wights_file) # load it
    NN_model.compile(optimizer = 'adam',loss='mae')

    NN_preds = NN_model.predict(X_test)

    NN_preds = np.asarray(NN_preds)
    np.save("predicted_rssi.npy",NN_preds)
    
    #score = NN_model.evaluate(X_test, Y_test, verbose=1)
    #print(NN_model.metrics,score)
    #print(Y_test)
    #print("predicting\n")
    #NN_preds = NN_model.predict(X_test)
    #print(NN_preds)
    #print("Shapes\n")
    #print(Y_test.shape, NN_preds.shape)
    #print("Absolute Difference\n")
    x = abs(Y_test-NN_preds)
    #print(x)
    x = np.asarray(x)
    np.save("loss.npy",x)
    print("Mean absolute Difference\n")
    #print(np.mean(x))
    '''
    plt.style.use('seaborn-whitegrid')

    figure(num=None, figsize=(10,5), dpi=100, facecolor='w', edgecolor='r')

    plt.plot(NN_preds, alpha = 0.75, color = 'blue')
    #plt.plot(Y_test, alpha = 0.60, color= 'red')
    
    plt.title('Predictions Values')

    plt.ylabel('RSSI')
    plt.xlabel('Test Value Number')

    plt.legend(['Predicted RSSI'], loc='upper left')

    plt.show()
    '''

def cleanup():
    from keras import backend as K
    
    K.clear_session()

    from numba import cuda
    cuda.select_device(0)
    cuda.close()
    
#train(NN_model)
test(NN_model)
#cleanup()
#print("mean Y_train",np.mean(Y_train))
#print("mean Y_test",np.mean(Y_test))
