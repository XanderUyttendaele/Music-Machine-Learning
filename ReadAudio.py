from scipy.io.wavfile import read
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

note = read("MAPS_data\\MAPS_AkPnBcht_1\\AkPnBcht\\ISOL\\NO\\MAPS_ISOL_NO_F_S0_M28_AkPnBcht.wav")
length = len(note[1])/note[0]
times = np.arange(0, length, 1/note[0])
frequencies = np.arange(0, note[0], note[0]/len(note[1]))
left = np.take(note[1], 0, axis=1)
right = np.take(note[1], 1, axis=1)
epsilon = 10**(-10)


def timePlot(input, times, title):
    plt.plot(times, input)
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()


timePlot(left, times, "Stereo Left")
timePlot(right, times, "Stereo Right")


def spectrogramPlot(input, title, fs, nperseg):
    f, t, Sxx = signal.spectrogram(input, fs=fs, nperseg=nperseg)
    plt.pcolormesh(t, f, 10*np.log10(Sxx + epsilon))
    plt.title(title)
    plt.ylabel("Frequency [hZ]")
    plt.xlabel("Time [s]")
    plt.show()


spectrogramPlot(left, "Left Spectrogram", note[0], 2048)
spectrogramPlot(right, "Right Spectrogram", note[0], 2048)


def fourierPlot(input, title):
    fft = np.fft.fft(input)
    plt.plot(frequencies, np.absolute(fft))
    plt.title(title + " Magnitude")
    plt.xlabel("Frequency [hZ]")
    plt.ylabel("Magnitude")
    plt.show()
    plt.plot(frequencies, np.angle(fft))
    plt.title(title + " Phase")
    plt.xlabel("Frequency [hZ]")
    plt.ylabel("Phase")
    plt.show()


fourierPlot(left, "Left FFT")
fourierPlot(right, "Right FFT")