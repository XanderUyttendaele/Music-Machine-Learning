"""
Code based on previous solution found on
http://zulko.github.io/blog/2014/02/12/transcribing-piano-rolls/
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

sample_file = "C:\\Users\\seb29\\OneDrive - Lakeside School\\Desktop\\Music Data\\MAPS_AkPnBcht_2\\AkPnBcht\\MUS\\MAPS_MUS-alb_se3_AkPnBcht"
#sample_file = "C:\\Users\\seb29\\OneDrive - Lakeside School\\Desktop\\Music Data\\MAPS_AkPnBcht_2\\AkPnBcht\\MUS\\MAPS_MUS-chpn_op66_AkPnBcht"
#sample_file = "C:\\Users\\seb29\\OneDrive - Lakeside School\\Desktop\\Music Data\\MAPS_AkPnBcht_2\\AkPnBcht\\MUS\\MAPS_MUS-ty_mai_AkPnBcht"

song = read(sample_file + ".wav")

song_length = 0
file = open(sample_file+".txt")
note_array = [[],[]]
for line in file:
    line_data = line.split()
    if(line_data[0] != "OnsetTime"):
        note_array[1].append(float(line_data[0]))
        note_array[0].append(int(line_data[2]) - 21)
        if(float(line_data[1]) > song_length):
            song_length = float(line_data[1])
        
file.close()
plt.plot(note_array[0],note_array[1])
plt.xlabel("Piano Key")
plt.ylabel("Onset Time")
plt.show()

keys_played = []
vals = []
for value in note_array[1]:
    if value not in vals:
        keys_played.append(note_array[1].count(value))
        vals.append(value)
        
plt.plot(vals, keys_played)
plt.xlabel("Onset Time")
plt.ylabel("Number of Keys")
plt.show()

fast_f = np.fft.fft(keys_played)

plt.plot(np.absolute(fast_f[1:]))
plt.xlabel("Frequency")
plt.ylabel("Spec Amp")
plt.show()

length_arr = np.arange(0,len(vals),song_length/len(vals))
max_val = np.argmax(np.absolute(fast_f[1:]))
beat_len = length_arr[max_val]/song_length
bpm = (60/beat_len)

mid_c = 60
left = []
right = []
for index in range(len(note_array[0])):
    if(note_array[1][index] < mid_c):
        left.append([note_array[1][index], note_array[0][index]])
    else:
        right.append([note_array[1][index], note_array[0][index]])


def midTreadQuantize(val, step):
    quantized = step * int(val/step + 1/2)
    return quantized

"""
We quantize our notes as per the specifications of the paper mentioned above:
    If d<Q/4, consider that the two notes belong to the same chord.
    Else, if Q/4≤d<3Q/4 , consider that the previous note was an eighth.
    Else, if 3Q/4≤d<5Q/4, consider that the previous note was a quarter
Where d represents the duration between two notes and Q represents the duration of a beat.
"""
def quantizeHand(hand_keys, beat_length):
    onset = [0]
    notes = [[]]
    note_length = [0]
    
    for index in range(len(hand_keys)):
        since_last_key = hand_keys[index][0] - hand_keys[index-1][0]
        #Quantize since_last_key/beat_len with Mid-Tread Quantizer, step size = 1/2
        since_last_quantized = midTreadQuantize(since_last_key/beat_length, 1/2)
        if(since_last_quantized != 0):
            note_length[-1] = since_last_quantized
            onset.append(hand_keys[index][0])
            notes.append([hand_keys[index][1]])
            #Duration of the final note is set to 0.0
            note_length.append(0.0)
        
        else:
            if hand_keys[index][1] not in notes[-1]:
                notes[-1].append(hand_keys[index][1])
    
    return onset[1:], notes[1:], note_length[1:]
    

left_q = quantizeHand(left, beat_len)
right_q = quantizeHand(right, beat_len)
    
print(left_q)
    
    
    
    
    
    
    
    
    
    
    