import pygame.midi
import time
timeStep = 512/22050.0
lengthToIgnore = 5 # ignore notes of this length or shorter
song_name = input("Song file name: ")
song = open("E:\\musicdata\\Music-Machine-Learning\\song_data\\" + song_name)
lineCount = 0
playArray = []
final = 0.0
for line in song:
    lineCount+=1
    if lineCount == 1:
        continue
    try:
        line_data = line.split()
        playArray.append([float(line_data[0]),float(line_data[1]),int(line_data[2])])
        if float(line_data[1]) > final:
            final = float(line_data[1])
    except:
        continue
curTime = 0.0
curPlayPos = 0
curFinishPos = 0
pygame.midi.init()
player = pygame.midi.Output(0)
player.set_instrument(0)
player.note_on(64, 127)
time.sleep(1)
player.note_off(64, 127)
volume = 127 # max 127

while(curTime < final):
    curTime+=timeStep
    on = []
    off = []
    for d in range(curPlayPos, len(playArray)):
        if playArray[d][0] <= curTime:
            on.append(playArray[d][2])
            curPlayPos+=1
        else:
            break
    for e in range(curFinishPos, len(playArray)):
        if playArray[d][1] <= curTime:
            off.append(playArray[d][2])
            curFinishPos+=1
        else:
            break
    for f in on:
        player.note_on(f,volume)
    for g in off:
        player.note_off(g, volume)
    time.sleep(timeStep)