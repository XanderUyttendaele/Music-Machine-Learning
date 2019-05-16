import tensorflow as tf
import numpy as np
import pygame.midi
import time
import pylab
import music21
import librosa
import math
import copy

# TODO: more precision? can make new numpy arrays with more precise values (smaller timestep) and feed through program
# DONE: move preprocess into this file so we can input wav directly
# TODO: Try different precision values
# DONE: Try to detect if the same note is being held but program is registering as separate notes - before
#  quantizing, check if there are a small amount of 0s (maybe continue a few more after finding some)
# TODO: Consider chord detection? Seems like certain notes are being dropped out early
# DONE: Fix 0 length notes
# DONE: Shift all offsets such that first note starts on beat one?
# TODO: Dynamics, use accents to figure out time signature and start note?
# TODO: If another note starts after another note starts and is still playing, invalidate previous note?
lengthToIgnore = 5 # ignore notes of this length or shorter - depending on length, can cause notes of length 0 after
# shifting
gapLength = 2
noteList = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
column_interval_sample = 512
timeStep = column_interval_sample/22050.0
frequency_bins = 252
bins_per_octave = 36
keyCount = 88
Fs = 22050
stepCount = int(math.floor(0.5/(column_interval_sample/22050.0)))
thresholdValue = 0.4 # recall?

def play():
    print("Playback started")
    pygame.midi.init()
    player = pygame.midi.Output(0)
    player.set_instrument(0)
    volume = 127  # max 127
    for b in range(length):
        on = []
        off = []
        for c in range(len(processedSongData)):
            if processedSongData[c][1] == b:
                on.append(processedSongData[c][0])
            if processedSongData[c][2] == b:
                off.append(processedSongData[c][0])
        for e in off:
            player.note_off(e + 21, volume)
        for d in on:
            player.note_on(d + 21, volume)
        time.sleep(timeStep)
    del player
    pygame.midi.quit()

def fourier_transform(signal, period, tt):
    f = lambda func : (signal*func(2*pylab.pi*tt/period)).sum()
    return f(pylab.cos)+ 1j*f(pylab.sin)

def closest(input, step):
    """
    Returns closest multiple of step to input
    """
    step = int(step)
    input = int(input)
    multiple = input//step
    input -= step*multiple
    if input > step / 2:
        multiple += 1
    return multiple * step


def threshold(i):
    if i > thresholdValue:
        return 1
    return 0

sess = tf.Session()
saver = tf.train.import_meta_graph("E:\\musicdata\\Music-Machine-Learning\\saved_models2\\biLSTM\\biLSTM256_50_4.0"
                                   ".ckpt.meta")
saver.restore(sess, "E:\\musicdata\\Music-Machine-Learning\\saved_models2\\biLSTM\\biLSTM256_50_4.0.ckpt")

print("Model restored.")
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
op = graph.get_tensor_by_name("prediction:0")
song_name = input("Song file name (wav): ")
try:
    components, rate = librosa.load(song_name)
except:
    print("Error loading file.")
song = np.abs(librosa.cqt(y = components, sr = rate, hop_length = column_interval_sample,n_bins = frequency_bins,
                         bins_per_octave = bins_per_octave)).transpose()
length = song.shape[0]

print("Song loaded.")
print("Begin processing:")
offset = length % stepCount
songData = [[0] * keyCount]*length
for i in range(length - stepCount + 1):
    feed_dict = {x:[song[i : i + stepCount]]}
    output = sess.run(op, feed_dict)
    for j in range(stepCount):
        songData[i + j] = [sum(x) for x in zip(songData[i+j], output[0][j])]
# for i in range(length//stepCount):
#     feed_dict = {x:[song[i * stepCount: (i+1) * stepCount]]}
#     output = sess.run(op, feed_dict)
#     for j in range(stepCount):
#         songData[i*stepCount + j] = output[0][j]
#     feed_dict = {x:[song[i * stepCount + offset: (i+1) * stepCount + offset]]}
#     output = sess.run(op, feed_dict)
#     for j in range(stepCount):
#         songData[i*stepCount + j + offset] = output[0][j]
# feed_dict = {x:[song[i * stepCount: (i+1) * stepCount]]}
# output = sess.run(op, feed_dict)
# for j in range(stepCount):
#     songData[-1 * (stepCount - j)] = output[0][j]
for i in range(length):
    if i < stepCount:
        songData[i] = [x / (i + 1) for x in songData[i]]
    elif i > length - stepCount:
        songData[i] = [x / (length - i) for x in songData[i]]
    else:
        songData[i] = [x / stepCount for x in songData[i]]
for i in range(length):
    songData[i] = [threshold(x) for x in songData[i]]
print("Done processing " + str(length) + " timesteps")
print("Converting to start/end format") # list of lists in format [note, start time, stop time]
processedSongData = []
unchangedSongData = copy.deepcopy(songData)
songData.append([False]*keyCount)
i = 0
j = 0
while i < length:
    for j in range(keyCount):
        if songData[i][j]:
            noteLength = 0
            iterator = i
            tryContinue = False
            while songData[iterator][j] or tryContinue:
                tryContinue = False # change to True to enable gap fixing
                if songData[iterator][j] or tryContinue:
                    if not songData[iterator][j]:
                        tryContinue = False
                    songData[iterator][j] = False
                    noteLength+=1
                    iterator+=1
            # code for fixing gaps, probably not a good idea (in tests, it messed up a lot of notes that should be
            # separate)
            # if noteLength != 0:
            #     oldNoteLength = noteLength
            #     for k in range(gapLength):
            #         if songData[iterator + gapLength - k][j]:
            #             for l in range(gapLength - k):
            #                 songData[iterator + l][j] = False
            #             noteLength += gapLength - k
            #             while songData[iterator + gapLength - k][j]:
            #                 songData[iterator + gapLength - k][j] = False
            #                 noteLength += 1
            #                 iterator += 1
            #             print("Fixed gap of length " + str(l) + " Added on " + str(noteLength-oldNoteLength) + " "
            #                                                                                                    "timesteps")
            #             break
            if noteLength >= lengthToIgnore:
                processedSongData.append([j,i, (i+noteLength)]) # * timeStep])
    i+=1
startCounts = []
endCounts = []
for b in range(length):
    on = []
    off = []
    for c in range(len(processedSongData)):
        if processedSongData[c][1] == b:
            on.append(processedSongData[c][0])
        if processedSongData[c][2] == b:
            off.append(processedSongData[c][0])
    startCounts.append(len(on))
    endCounts.append(len(off))
# playback = input("Play predictions? (y/n)")
# if playback == 'y':
#     play()
print("Beginning beat detection")
tt = pylab.arange(len(startCounts))
durations = pylab.arange(1.1,30,.02) # avoid 1.0
transform = pylab.array([fourier_transform(startCounts,d, tt) for d in durations])
# # not working sometimes, maybe because many large values close to spike?
# spikeCount = len([i for i in abs(transform) if i > 0.75 * pylab.argmax(abs(transform))])
# if spikeCount == 1:
#     top_k = np.array([pylab.argmax(abs(transform)),pylab.argmax(abs(transform))])
# else:
#     top_k = pylab.argpartition(abs(transform), -2)[-2:]
#     top_k = np.sort(top_k)
# # should be getting the larger of the 2
# quarter_duration = int(round(durations[top_k[1]]))
precision = 2 #int(round(top_k[1]/top_k[0])) # how many parts to split a quarter note into
quarter_duration = int(round(durations[pylab.argmax(abs(transform))]))
if 60.0/(quarter_duration * timeStep) > 180:
    quarter_duration *= 2
    precision *= 2
elif 60.0/(quarter_duration * timeStep) < 60:
    quarter_duration /= 2
    precision /= 2
# won't always be 2, should be 3 in some cases, but not gonna implement that atm
precision = int(precision)
pylab.plot(durations, abs(transform))
pylab.xlabel('period (in timesteps)')
pylab.ylabel('Spectrum value')
# pylab.show()
print("Estimated tempo (bpm): " + '%.3f' % (60.0/(quarter_duration * timeStep)))

print("Shifting notes to tempo")
print("Shifting notes to 1/" + str(precision) + " of a quarter")
length *= precision
timeStep /= precision
shortestDuration = quarter_duration
quarter_duration *= precision
processedSongData = [[note[0],note[1]*precision,note[2]*precision]for note in processedSongData]
shift = 0
for note in processedSongData:
    temp = note[1]
    temptwo = note[2]
    if note[1] % (shortestDuration) > 3 * (shortestDuration) / 4:
        note[1] += shortestDuration - (note[1] % shortestDuration)
    else:
        note[1] -= note[1] % shortestDuration
    if note[2] % shortestDuration < shortestDuration / 4:
        note[2] -= note[2] % shortestDuration
    else:
        note[2] += shortestDuration - (note[2] % shortestDuration)
    # note[2] = note[1] + shortestDuration # set all notes to shortest length
    if note[2] - note[1] > 4 * shortestDuration: # anything larger than this doesn't usually happen (e.g. a note
    # with length of five sixteenths
        if precision == 4 and note[2] - note[1] <= 8 * shortestDuration: # allowing dotted quarters (only regular
            # note without quarter note multiple length
            note[2] = note[1] + closest(note[2] - note[1], 2 * shortestDuration)
        else:
            note[2] = note[1] + closest(note[2] - note[1], quarter_duration)
    if note[2] - note[1] >= 4 * shortestDuration:
        if not int(round(note[1] / shortestDuration)) % 2 == 1:
            note[1] -= shortestDuration
            note[2] -= shortestDuration
    if note[2] == note[1]:
        note[2] += shortestDuration
        print("Fixed 0 length note " + str(note[0]) + " at " + str(note[1]/(quarter_duration)))
    shift += note[1] - temp
    shift += note[2] - temptwo
    shift += note[2] - temp
shift /= (len(processedSongData)*2)
shift *= timeStep
print("Average shift (seconds): " + '%.3f' % shift)
# make sure they're not overlapping
lastPosNotes = [0]*88
for note in processedSongData:
    if lastPosNotes[note[0]] > note[1]:
        note[1] = lastPosNotes[note[0]]
    lastPosNotes[note[0]] = note[2]
playback = input("Play predictions? (y/n)")
if playback == 'y':
    play()
print("Beginning sheet music generation - generating notes")
tempoMarking = music21.tempo.MetronomeMark(number=int(60.0/(quarter_duration * timeStep)))
score = music21.stream.Score()
right = [music21.stream.Part()]
curRight = 0
left = [music21.stream.Part()]
curLeft = 0
least = 0
for a in range(length):
    for c in range(len(processedSongData)):
        if processedSongData[c][1] == a:
            # shift all notes so first note is at 0
            if least == 0:
                least = a
            processedSongData[c][2] -= least
            processedSongData[c][1] -= least
            octave = (processedSongData[c][0]+9)//12
            temp = noteList[processedSongData[c][0] + 9 - octave * 12]
            toAppend = music21.note.Note(temp+str(octave))
            toAppend.quarterLength = (processedSongData[c][2]-processedSongData[c][1])/quarter_duration
            if processedSongData[c][0]>=39:
                if curRight >= len(right):
                    right.append(music21.stream.Part())
                right[curRight].insert(processedSongData[c][1]/quarter_duration,toAppend)
                curRight+=1
            else:
                if curLeft >= len(left):
                    left.append(music21.stream.Part())
                left[curLeft].insert(processedSongData[c][1]/quarter_duration,toAppend)
                curLeft+=1
        if processedSongData[c][2] == a:
            if processedSongData[c][0]>=39:
                curRight-=1
            else:
                curLeft-=1
rightScore = music21.stream.Score()
leftScore = music21.stream.Score()
for part in right:
    rightScore.insert(0, part)
for part in left:
    leftScore.insert(0, part)
# making them all end with full measures
rightLength = rightScore.quarterLength
leftLength = leftScore.quarterLength
rightRest = music21.note.Rest()
rightRest.duration.quarterLength = 4.0 - rightLength % 4
rightScore.append(rightRest)
leftRest = music21.note.Rest()
leftRest.duration.quarterLength = 4.0 - leftLength % 4
leftScore.append(leftRest)
while leftScore.quarterLength > rightScore.quarterLength:
    fullRest = music21.note.Rest()
    fullRest.duration.quarterLength = 4.0
    rightScore.append(fullRest)
while rightScore.quarterLength > leftScore.quarterLength:
    fullRest = music21.note.Rest()
    fullRest.duration.quarterLength = 4.0
    leftScore.append(fullRest)
score.append(tempoMarking)
score.insert(0, rightScore.chordify())
score.insert(0, leftScore.chordify())
keySig = score.analyze('key')
score.keySignature = keySig
score.timeSignature = music21.meter.TimeSignature("4/4") # change based on something? idk
score.show()