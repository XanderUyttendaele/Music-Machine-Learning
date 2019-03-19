import numpy as np
from scipy.io.wavfile import read
import matplotlib.pyplot as plt


NOTES_IN_SCALE = 12
F_REF = 440
Fs = 44100
NOTES = ['A', 'A#/Bb', 'B', 'C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab']


def calculate_PCP(note):
    N = len(note)
    window = np.hamming(N)
    windowed_note = [note[i]*window[i] for i in range(N)]
    fft = np.absolute(np.fft.fft(windowed_note))
    PCP = [0 for i in range(NOTES_IN_SCALE)]
    M = np.round(12*np.log(Fs/F_REF*np.arange(1,N/2)/N)/np.log(2)) % NOTES_IN_SCALE
    for i in range(M.size):
        PCP[int(M[i])] += fft[i+1]**2
    return PCP


def calculate_chromagram(file, frame_width = 512):
    file = np.mean(read(file)[1], axis = 1)
    num_frames = int(2*file.size/frame_width - 1)
    chromagram = np.zeros((num_frames, NOTES_IN_SCALE))
    for i in range(num_frames):
        PCP = calculate_PCP(file[int(i*frame_width/2):int(i*frame_width/2+frame_width)])
        norm = np.linalg.norm(PCP)
        if norm != 0:
            PCP_normalized = [i/np.linalg.norm(PCP) for i in PCP]
        else:
            PCP_normalized = PCP
        chromagram[i] = PCP_normalized
    return chromagram


def interpret_PCP(PCP):
    PCP_scaled = [i/np.max(PCP) for i in PCP]
    notes = [i > 0.1 for i in PCP_scaled]
    note_string = ""
    for i in range(0, NOTES_IN_SCALE):
        if notes[i]:
            if i != 0:
                note_string += ", "
            note_string += NOTES[i]
    return note_string


def plot_chromagram(chromagram):
    plot = plt.pcolormesh(chromagram.transpose())
    plt.colorbar(plot)
    plt.show()