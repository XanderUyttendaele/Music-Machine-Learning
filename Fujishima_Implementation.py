import numpy as np

NOTES_IN_SCALE = 12
F_REF = 440
Fs = 44100
NOTES = ['A', 'A#/Bb', 'B', 'C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab']


def detect_pitches(note):
    fft = np.absolute(np.fft.fft(note))
    N = len(note)
    PCP = [0 for i in range(NOTES_IN_SCALE)]
    M = np.round(12*np.log(Fs/F_REF*np.arange(1,N/2)/N)/np.log(2)) % NOTES_IN_SCALE
    for i in range(M.size):
        PCP[int(M[i])] += fft[i+1]**2
    return PCP


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
