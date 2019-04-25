import librosa
import numpy as np
import os
import os.path as pt
import math

column_interval_sample = 512
frequency_bins = 252
bins_per_octave = 36
num_notes = 88
Fs = 22050

def convert_to_array():
    count = 0
    for root, directories, filenames in os.walk("E:\\musicdata\\Music-Machine-Learning\\song_data\\"):
        for filename in filenames:
            file_path = pt.join(root, filename)
            base_path = pt.splitext(pt.basename(pt.normpath(file_path)))[0]
            if(not pt.isfile("E:\\musicdata\\Music-Machine-Learning\\song_data_labeled\\" + base_path + ".npy")):

                #Produce Constant Q Spectrogram
                components, rate = librosa.load(root+"\\"+base_path+".wav")
                const_q_vals = np.abs(librosa.cqt(y = components, sr = rate, hop_length = column_interval_sample, n_bins = frequency_bins, bins_per_octave = bins_per_octave)).transpose() #Determine Constant Q values as per our above specifications.

                np.save("E:\\musicdata\\Music-Machine-Learning\\song_data_training\\" + base_path, const_q_vals)
                #Each chord label will be represented by the frames of the spectrogram and the number of possible notes
                num_spec_cols = const_q_vals.shape[0]
                sampling_vec = column_interval_sample*np.arange(num_spec_cols)/float(rate)
                note_label = np.zeros((num_spec_cols,num_notes))

                #Load our label array with provided file data
                note_file = open(root+"\\"+base_path+".txt")
                for line in note_file:
                    line_data = line.split()
                    try:
                        if line_data[0] != "OnsetTime":
                            begin = float(line_data[0])
                            end = float(line_data[1])
                            mini = 0
                            maxi = len(sampling_vec)-1
                            while(begin > sampling_vec[mini]):
                                mini+= 1
                            while(end < sampling_vec[maxi]):
                                maxi-= 1
                            note_pitch = int(line_data[2]) - 21
                            note_label[mini:maxi, note_pitch] = 1
                    except:
                        pass
                note_file.close()
                if(not pt.isfile("E:\\musicdata\\Music-Machine-Learning\\song_data_labeled\\" + base_path + ".npy")):
                    np.save("E:\\musicdata\\Music-Machine-Learning\\song_data_labeled\\" + base_path, note_label)

            count+=1
            print(count)
            if(count%100 == 0):
                print(count)


def segment_files(length):
    step_count = int(math.floor(length / (column_interval_sample / Fs)))  # length seconds, assuming 512 hop length and 22050 rate
    for root, directories, filenames in os.walk("song_data/song_data_training"):
        for filename in filenames:
            file_path = pt.join(root, filename)
            base_path = pt.splitext(pt.basename(pt.normpath(file_path)))[0]
            song = np.load("song_data/song_data_training/" + filename)
            label = np.load("song_data/song_data_labeled/" + filename)
            clips = int(math.floor(song.shape[0] / step_count))
            for i in range(clips):
                song_segment = song[i*step_count:(i+1)*step_count]
                np.save("song_data/song_data_training_shortened/" + base_path + "_" + str(i) + ".npy", song_segment)
                label_segment = label[i*step_count:(i+1)*step_count]
                np.save("song_data/song_data_labeled_shortened/" + base_path + "_" + str(i) + ".npy", label_segment)
