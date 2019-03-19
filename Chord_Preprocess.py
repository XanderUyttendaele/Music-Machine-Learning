import librosa
import numpy as np
import os
import os.path as pt

column_interval_sample = 512
frequency_bins = 252
bins_per_octave = 36
num_notes = 88

count = 0
for root, directories, filenames in os.walk("C:\\Users\\seb29\\OneDrive - Lakeside School\\Desktop\\Music Data\\AkPnBcht\\UCHO\\"):
    for filename in filenames:
        file_path = pt.join(root, filename)
        base_path = pt.splitext(pt.basename(pt.normpath(file_path)))[0]        
        
        #Produce Constant Q Spectrogram
        components, rate = librosa.load(root+"\\"+base_path+".wav")
        const_q_vals = np.abs(librosa.cqt(y = components, sr = rate, hop_length = column_interval_sample, n_bins = frequency_bins, bins_per_octave = bins_per_octave)) #Determine Constant Q values as per our above specifications.
        if(not pt.isfile("C:\\Users\\seb29\\OneDrive - Lakeside School\\Desktop\\chord_labeled_data\\" + base_path + ".npy")):
            np.save("C:\\Users\\seb29\\OneDrive - Lakeside School\\Desktop\\chord_training_data\\" + base_path, const_q_vals)
        
        #Each chord label will be represented by the frames of the spectrogram and the number of possible notes
        num_spec_cols = const_q_vals.shape[1]
        sampling_vec = column_interval_sample*np.arange(num_spec_cols)/float(rate)
        note_label = np.zeros((num_spec_cols,num_notes))
        
        #Load our label array with provided file data
        note_file = open(root+"\\"+base_path+".txt")
        for line in note_file:
            line_data = line.split()
            if(line_data[0] != "OnsetTime"):
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
        note_file.close()
        if(not pt.isfile("C:\\Users\\seb29\\OneDrive - Lakeside School\\Desktop\\chord_labeled_data\\" + base_path + ".npy")): 
            np.save("C:\\Users\\seb29\\OneDrive - Lakeside School\\Desktop\\chord_labeled_data\\" + base_path, note_label)
        
        count+=1
        if(count%100 == 0):
            print(count)