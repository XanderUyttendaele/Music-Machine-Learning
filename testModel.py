import tensorflow as tf
import numpy as np
import random
import pygame.midi
import time

timeStep = 512/22050.0
lengthToIgnore = 4 # ignore notes of this length or shorter

songData = []

sess = tf.Session()
saver = tf.train.import_meta_graph("E:\\musicdata\\Music-Machine-Learning\\saved_models\\model.ckpt.meta")
saver.restore(sess, tf.train.latest_checkpoint("E:\\musicdata\\Music-Machine-Learning\\saved_models\\"))

print("Model restored.")
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
op = graph.get_tensor_by_name("prediction:0")
song_name = input("Song file name: ")
song = np.load("E:\\musicdata\\Music-Machine-Learning\\song_data_training\\" + song_name)
length = song.shape[0]
labels = np.load("E:\\musicdata\\Music-Machine-Learning\\song_data_labeled\\" + song_name)
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
playback = input("Play predictions? (y/n)")
if playback == 'y':
    playback = True
    print("Playback started")
    pygame.midi.init()
    player = pygame.midi.Output(0)
    player.set_instrument(0)
    player.note_on(64, 127)
    time.sleep(1)
    player.note_off(64, 127)
    volume = 127 # max 127
else:
    playback = False
for b in range(length):
    on = []
    off = []
    for c in range(len(processedSongData)):
        if processedSongData[c][1] == b:
            on.append(processedSongData[c][0])
        if processedSongData[c][2] == b:
            off.append(processedSongData[c][0])
    if playback:
        for e in off:
            player.note_off(e+21,volume)
        for d in on:
            player.note_on(d+21,volume)
    time.sleep(timeStep)
    startCounts.append(len(on))
    endCounts.append(len(off))
if playback:
    del player
    pygame.midi.quit()

