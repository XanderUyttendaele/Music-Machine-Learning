from flask import Flask, request, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import pygame.midi
import time
import pylab
import music21
import librosa
import math
import copy
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXTENSIONS = set(["WAV"])
lengthToIgnore = 5
gapLength = 2
noteList = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
column_interval_sample = 512
timeStep = column_interval_sample/22050.0
frequency_bins = 252
bins_per_octave = 36
keyCount = 88
Fs = 22050
stepCount = int(math.floor(0.5/(column_interval_sample/22050.0)))
thresholdValue = 0.4
model_root = app.config["UPLOAD_FOLDER"]
model_name = "\\model"

def play():
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

@app.route("/converter", methods = ['GET', 'POST'])
def song_upload():
	if request.method == "POST":
		if "song" not in request.files:
			return redirect("http://www.1819.lakeside-cs.org/MusicML/index.html")
		song_file = request.files['song']
		if song_file and (song_file.filename.rsplit(".",1)[1].upper() in ALLOWED_EXTENSIONS):
			global timeStep
		
			#Saving temporary file
			file_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(song_file.filename))
			song_file.save(file_path)
			
			#Initiating tensorflow model
			sess = tf.Session()
			saver = tf.train.import_meta_graph(model_root+model_name+".ckpt.meta")
			saver.restore(sess, model_root+model_name+".ckpt")
			
			#Setting graph for processing
			graph = tf.get_default_graph()
			x = graph.get_tensor_by_name("x:0")
			op = graph.get_tensor_by_name("prediction:0")
			components, rate = librosa.load(file_path)
			song = np.abs(librosa.cqt(y = components, sr = rate, hop_length = column_interval_sample,n_bins = frequency_bins,bins_per_octave = bins_per_octave)).transpose()
			
			#Processing
			length = song.shape[0]
			songData=[]
			bottomNotesCorrect = 0
			bottomNotesTotal = 0
			for i in range(length):
				feed_dict = {x: [[song[i] for a in range(21)]]}
				songData.append(sess.run(op, feed_dict)[0])
			
			#Converting to start/end format
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
						if noteLength >= lengthToIgnore:
							processedSongData.append([j,i, (i+noteLength)])
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
			
			#Beat detection
			tt = pylab.arange(len(startCounts))
			durations = pylab.arange(1.1,30,.02) # avoid 1.0
			transform = pylab.array([fourier_transform(startCounts,d, tt) for d in durations])
			precision = 2
			quarter_duration = int(round(durations[pylab.argmax(abs(transform))]))
			if 60.0/(quarter_duration * timeStep) > 180:
				quarter_duration *= 2
				precision *= 2
			elif 60.0/(quarter_duration * timeStep) < 60:
				quarter_duration /= 2
				precision /= 2
			precision = int(precision)
			
			#Shifting notes
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
				if note[2] - note[1] > 4 * shortestDuration:
					if precision == 4 and note[2] - note[1] <= 8 * shortestDuration:
						note[2] = note[1] + closest(note[2] - note[1], 2 * shortestDuration)
					else:
						note[2] = note[1] + closest(note[2] - note[1], quarter_duration)
				if note[2] - note[1] >= 4 * shortestDuration:
					if not int(round(note[1] / shortestDuration)) % 2 == 1:
						note[1] -= shortestDuration
						note[2] -= shortestDuration
				if note[2] == note[1]:
					note[2] += shortestDuration
				shift += note[1] - temp
				shift += note[2] - temptwo
				shift += note[2] - temp
			shift /= (len(processedSongData)*2)
			shift *= timeStep
			lastPosNotes = [0]*88
			for note in processedSongData:
				if lastPosNotes[note[0]] > note[1]:
					note[1] = lastPosNotes[note[0]]
				lastPosNotes[note[0]] = note[2]
			
			#Generating sheet music
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
			score.timeSignature = music21.meter.TimeSignature("4/4")
			song_name = secure_filename(song_file.filename).rsplit(".",1)[0]
			score.insert(0,music21.metadata.Metadata())
			score.metadata.title = song_name.rsplit("\\")[-1]
			score.write("xml", fp = os.path.join(app.config["UPLOAD_FOLDER"], song_name+".xml"))
			
			os.remove(file_path)
				
			return send_from_directory(app.config["UPLOAD_FOLDER"], secure_filename(song_file.filename).rsplit(".",1)[0]+".xml", as_attachment=True)
	return redirect("http://www.1819.lakeside-cs.org/MusicML/index.html")
		

	



