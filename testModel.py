import tensorflow as tf
import random
import pygame.midi
import time
import pylab
import music21

# TODO: more precision? can make new numpy arrays with more precise values (smaller timestep) and feed through program
# TODO: move preprocess into this file so we can input wav directly
timeStep = 512/22050.0
lengthToIgnore = 5 # ignore notes of this length or shorter - depending on length, can cause notes of length 0 after
# shifting
noteList = ["C","C#","D","D#","E","E#","F","G","G#","A","A#","B"]

songData = []

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
    """ See http://en.wikipedia.org/wiki/Fourier_transform
    I could also have used Numpy's fft.
    """
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

sess = tf.Session()
saver = tf.train.import_meta_graph("E:\\musicdata\\Music-Machine-Learning\\saved_models\\model.ckpt.meta")
saver.restore(sess, tf.train.latest_checkpoint("E:\\musicdata\\Music-Machine-Learning\\saved_models\\"))

print("Model restored.")
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
op = graph.get_tensor_by_name("prediction:0")
song_name = input("Song file name: ")
song = pylab.load("E:\\musicdata\\Music-Machine-Learning\\song_data_training\\" + song_name)
length = song.shape[0]
labels = pylab.load("E:\\musicdata\\Music-Machine-Learning\\song_data_labeled\\" + song_name)
keyCount = labels.shape[1]
print("Song loaded.")
print("Begin processing:")
bottomNotesCorrect = 0
bottomNotesTotal = 0
for i in range(length):
    feed_dict = {x:[[song[i] for a in range(21)]]}
    songData.append(sess.run(op, feed_dict)[0])
    for j in range(keyCount):
        if labels[i][j]:
            if songData[i][j]:
                bottomNotesCorrect += 1
            bottomNotesTotal += 1
            break
print("Done processing " + str(length) + " timesteps")
print("Bottom note precision: " + str(bottomNotesCorrect) + " / " + str(bottomNotesTotal))
print("Picking random test")

index = random.randint(0, length-1)
print("Timestep: " + str(index))
print("Actual notes: " + str([i for i, e in enumerate(labels[index].tolist()) if e == 1.0]))
print("Predictions: " + str([i for i, e in enumerate(songData[index]) if e]))
time.sleep(5)
print("Converting to start/end format") # list of lists in format [note, start time, stop time]
processedSongData = []
songData.append([False]*keyCount)
i = 0
j = 0
while i < length:
    for j in range(keyCount):
        if songData[i][j]:
            noteLength = 0
            iterator = i
            while songData[iterator][j]:
                songData[iterator][j] = False
                noteLength+=1
                iterator+=1
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
playback = input("Play predictions? (y/n)")
if playback == 'y':
    play()
print("Beginning beat detection")
tt = pylab.arange(len(startCounts))
durations = pylab.arange(1.1,30,.02) # avoid 1.0
transform = pylab.array([fourier_transform(startCounts,d, tt)
                    for d in durations] )
optimal_i = pylab.argmax(abs(transform))
quarter_duration = int(durations[optimal_i])

pylab.plot(durations, abs(transform))
pylab.xlabel('period (in timesteps)')
pylab.ylabel('Spectrum value')
pylab.show()
print("Estimated tempo (bpm): " + '%.3f' % (60.0/(quarter_duration * timeStep)))

print("Shifting notes to tempo")
precision = 4 # how many parts to split a quarter note into
length *= precision
timeStep /= precision
quarter_duration *= precision
processedSongData = [[note[0],note[1]*precision,note[2]*precision]for note in processedSongData]
shift = 0
for note in processedSongData:
    temp = note[1]
    note[1] = closest(note[1], quarter_duration/precision)
    # int((note[1]//(quarter_duration/precision))*(quarter_duration/precision)) does it make more sense to use closest or just floor - if there's
    # a note in the middle of two 16ths or w/e do we assume it started at the previous 16th?
    shift += note[1] - temp
    temp = note[2]
    note[2] = closest(note[2], quarter_duration/precision)
    # int((note[2]//(quarter_duration/precision))*(quarter_duration/precision))
    shift += note[2] - temp
shift /= (len(processedSongData)*2)
shift *= timeStep
print("Average shift (seconds): " + '%.3f' % shift)
playback = input("Play predictions? (y/n)")
if playback == 'y':
    play()
print("Beginning sheet music generation - generating notes")
score = music21.stream.Score()
right = [music21.stream.Part()]
curRight = 0
left = [music21.stream.Part()]
curLeft = 0
for a in range(length):
    for c in range(len(processedSongData)):
        if processedSongData[c][1] == a:
            octave = (processedSongData[c][0]+9)//12
            temp = noteList[processedSongData[c][0] + 9 - octave * 12]
            toAppend = music21.note.Note(temp+str(octave))
            toAppend.quarterLength = (processedSongData[c][2]-processedSongData[c][1])/quarter_duration
            if processedSongData[c][0]>=39:
                if curRight >= len(right):
                    right.append(music21.stream.Part())
                right[curRight].append(toAppend)
                right[curRight][-1].offset = processedSongData[c][1]/quarter_duration
                curRight+=1
            else:
                if curLeft >= len(left):
                    left.append(music21.stream.Part())
                left[curLeft].append(toAppend)
                left[curLeft][-1].offset = processedSongData[c][1]/quarter_duration
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
score.insert(0, rightScore.chordify())
score.insert(0, leftScore.chordify())
score.show()