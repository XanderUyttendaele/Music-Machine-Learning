import matplotlib.pyplot as plt
import librosa
import numpy as np


def generateSpec(column_interval_sample, frequency_bins, bins_per_octave, file_path):
    components, rate = librosa.load(file_path)
    const_q_vals = np.abs(librosa.cqt(y = components, sr = rate, hop_length = column_interval_sample, n_bins = frequency_bins, bins_per_octave = bins_per_octave)).transpose() #Determine Constant Q values as per our above specifications. 
    
    plt.figure()
    plt.imshow(np.abs(const_q_vals), aspect = "auto")
    plt.plot()
