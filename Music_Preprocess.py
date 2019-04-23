import librosa
import numpy as np
import os
import os.path as pt

column_interval_sample = 512
frequency_bins = 252
bins_per_octave = 36
num_notes = 88
data_path = "C:\\Users\\seb29\\OneDrive - Lakeside School\\Desktop\\Music Data\\"

zeros = np.zeros(252)
portion_size = 10


for directory in os.scandir(data_path):
    if directory.name.endswith("2"):
        for root, directories, filenames in os.walk(data_path+directory.name+"\\"+directory.name[5:-2]+"\\"):
            for filename in filenames:
                file_path = pt.join(root, filename)
                base_path = pt.splitext(pt.basename(pt.normpath(file_path)))[0]  
                
                portion_start = 0
                index = 1
                portion_end = portion_size
                components, rate = librosa.load(path = root+"\\"+base_path+".wav")
                note_file = open(root+"\\"+base_path+".txt")
                
                song_text = []
                for line in note_file:
                    line_data = line.split()
                    if(line_data[0] != "OnsetTime"):
                        begin = float(line_data[0])
                        end = float(line_data[1])
                        note_pitch = int(line_data[2]) - 21 
                        song_text.append([begin, end, note_pitch])
                note_file.close()
                
                #Produce Constant Q Spectrograms in segments of each song
                while(portion_end < librosa.get_duration(components)):
                    portion = components[portion_start*rate:portion_end*rate]
                    const_q_vals = np.abs(librosa.cqt(y = portion, sr = rate, hop_length = column_interval_sample, n_bins = frequency_bins, bins_per_octave = bins_per_octave)).transpose() #Determine Constant Q values as per our above specifications.
                    np.save("C:\\Users\\seb29\\OneDrive - Lakeside School\\Desktop\\music_training_data\\" + base_path + "_"+str(portion_start), const_q_vals)

                    #Each chord label will be represented by the frames of the spectrogram and the number of possible notes
                    num_spec_cols = const_q_vals.shape[0]
                    sampling_vec = column_interval_sample*np.arange(num_spec_cols)/float(rate)
                    note_label = np.zeros((num_spec_cols,num_notes))

                    while(index < len(song_text) and portion_end > song_text[index][0]):
                        mini = 0
                        maxi = len(sampling_vec)-1
                        while(mini < len(sampling_vec) and song_text[index][0]-portion_start > sampling_vec[mini]):
                            mini+=1
                        while(maxi < len(sampling_vec) and song_text[index][1]-portion_start < sampling_vec[maxi]):
                            maxi-=1
                        note_label[mini:maxi, note_pitch] = 1
                        index+=1
                    np.save("C:\\Users\\seb29\\OneDrive - Lakeside School\\Desktop\\music_labeled_data\\" + base_path + "_"+str(portion_start), note_label)
        
                    portion_start = portion_end
                    portion_end = portion_end + portion_size