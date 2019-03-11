import numpy as np

NOTES_IN_SCALE = 12
F_REF = 440
Fs = 44100
NOTES = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']


def detect_pitches(note):
    fft = np.absolute(np.fft.fft(note))
    N = len(note)
    PCP = [0 for i in range(NOTES_IN_SCALE)]
    M = np.round(12*np.log(Fs/F_REF*np.arange(1,N)/N)/np.log(2)) % NOTES_IN_SCALE
    for i in range(M.size):
        PCP[int(M[i])] += fft[i]**2
    return PCP


def interpret_PCP(PCP):
    return NOTES[np.argmax(PCP)]
