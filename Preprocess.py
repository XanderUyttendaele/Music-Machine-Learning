import glob, os
import numpy as np
from scipy.io.wavfile import read
import pandas as pd

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
    return features, targets