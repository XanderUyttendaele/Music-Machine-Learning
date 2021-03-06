import librosa
import numpy as np
import os
import os.path as pt

column_interval_sample = 128
frequency_bins = 252
bins_per_octave = 36
num_notes = 88

count = 0
for root, directories, filenames in os.walk("E:\\musicdata\\Music-Machine-Learning\\staccato_notes_no_sustain\\"):
    for filename in filenames:
        file_path = pt.join(root, filename)
        base_path = pt.splitext(pt.basename(pt.normpath(file_path)))[0]        
        
        #Produce Constant Q Spectrogram
        components, rate = librosa.load(root+"\\"+base_path+".wav")
        const_q_vals = np.abs(librosa.cqt(y = components, sr = rate, hop_length = column_interval_sample, n_bins = frequency_bins, bins_per_octave = bins_per_octave)).transpose() #Determine Constant Q values as per our above specifications.
        
        
        #Each note label will be represented by the frames of the spectrogram and the number of possible notes
        num_spec_cols = const_q_vals.shape[0]
        print(num_spec_cols)
        sampling_vec = column_interval_sample * np.arange(num_spec_cols)/(float(rate))
        note_label = np.zeros(num_notes)
        
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
                note_label[note_pitch] = 1
        note_file.close()

        # if(not pt.isfile("E:\\musicdata\\Music-Machine-Learning\\staccato_labeled_data\\" + base_path + ".npy")):
        #     np.save("E:\\musicdata\\Music-Machine-Learning\\staccato_training_data\\" + base_path, const_q_vals[mini:maxi])
        # if(not pt.isfile("E:\\musicdata\\Music-Machine-Learning\\staccato_labeled_data\\" + base_path + ".npy")): 
        #     np.save("E:\\musicdata\\Music-Machine-Learning\\staccato_labeled_data\\" + base_path, note_label)
        print("{0}".format(const_q_vals[mini:maxi].shape))
        print("{0}".format(note_label.shape))
        
        count+=1
        if(count%100 == 0):
            print(count)