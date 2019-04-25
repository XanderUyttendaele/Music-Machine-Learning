import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
import pandas as pd
import os
import os.path as pt
import random
import matplotlib.pyplot as plt

batch_size = 20
iterations = 10
epoch_size = 5000000

training_dir = "C:\\Users\\seb29\\OneDrive - Lakeside School\\Desktop\\music_training_data\\"
target_dir = "C:\\Users\\seb29\\OneDrive - Lakeside School\\Desktop\\music_labeled_data\\"
feature_data = pd.Series()
target_data = pd.Series()
data = pd.DataFrame()
example_num = 0
num_files = len([filename for filename in os.listdir(training_dir) if os.path.isfile(os.path.join(training_dir, filename))])
timesteps = np.load(training_dir + random.choice(os.listdir(training_dir))).shape[0]
feat_num = 252
freq_num = 88

def getData():
    print("Loading Data:")
    feature_data = np.zeros([num_files, timesteps, feat_num])
    target_data = np.zeros([num_files, timesteps, freq_num])
    
    f_num = 0
    for root, directories, filenames in os.walk(training_dir):
        for filename in filenames:
            feature_data[f_num] = np.load(pt.join(root, filename))
            target_data[f_num] = np.load(pt.join(target_dir, filename))
            f_num += 1
            if f_num%100 == 0:
                print(">", end ="")
    print("\n\n")
            
    index = np.random.permutation(num_files)
    feature_data = feature_data[index]
    target_data = target_data[index]
    
    #ignoring test data for now
    train_data = feature_data[:batch_size*iterations]
    train_targets = target_data[:batch_size*iterations]
    #Note validation is shortened here for testing
    validation_data = feature_data[batch_size*iterations:batch_size*iterations+ iterations]
    validation_targets = target_data[batch_size*iterations:batch_size*iterations+ iterations]
    
    return train_data, train_targets, validation_data, validation_targets

def trainModel(train_data, train_targets, validation_data, validation_targets):
    model = keras.Sequential()
    model.add(Conv1D(252, kernel_size = 6, strides = 1, activation = "relu", input_shape = (431, 252)))
    model.add(MaxPooling1D(pool_size = 2, strides = 1))
    model.add(Conv1D(400, kernel_size = 6, activation = "relu"))
    model.add(MaxPooling1D(pool_size = 2))
    model.add(Flatten())
    model.add(Dense(500, activation = "relu"))
    model.add(Dense(500, activation = "relu"))
    model.add(Dense(37928, activation = "sigmoid"))
    model.compile(loss = "categorical_crossentropy", optimizer = keras.optimizers.SGD(lr = .01), metrics = [keras.metrics.categorical_accuracy])
    
    hist = model.fit(train_data, train_targets, validation_data = (validation_data, validation_targets), epochs = epoch_size, batch_size = batch_size, verbose = 1)
    showHistory(hist)

"""
Method taken from Keras tutorials with slight modificaitons.

"""
def showHistory(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

train_d, train_t, validation_d, validation_t = getData()
train_t = train_t.reshape(train_t.shape[:-2] +(-1,))
validation_t = validation_t.reshape(validation_t.shape[:-2] +(-1,))
trainModel(train_d, train_t, validation_d, validation_t)