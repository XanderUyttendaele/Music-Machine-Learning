import glob, os
import numpy as np
from scipy.io.wavfile import read
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import Dataset


Fs = 44100
onset = 0.5
offset = 2.5


def preprocess_features():
    os.chdir("MAPS_data\\MAPS_AkPnBcht_1\\AkPnBcht\\ISOL\\NO")
    files = glob.glob("*.wav")
    filelength = int(Fs*(offset-onset))
    features = np.zeros((len(files), filelength))
    targets = np.zeros(len(files))
    for i in range(len(files)):
        features[i] = np.absolute(np.fft.fft(np.take(read(files[i])[1][int(Fs*onset):int(Fs*offset)], 0, axis=1)))
        targets[i] = files[i].split("_")[5][1:]
    features = pd.DataFrame(features)
    targets = pd.DataFrame(targets)
    for i in range(0, 5):
        os.chdir("..")
    return features, targets  # Don't forget to shuffle at some point.


# Edited from machine learning crash course
def construct_feature_columns(input_features):
    return set([tf.feature_column.numeric_column(str(my_feature)) for my_feature in input_features])


# Copied from machine learning crash course
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key: np.array(value) for key, value in dict(features).items()}
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds = ds.shuffle(10000)
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
