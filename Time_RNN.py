import os, glob
import numpy as np
from scipy.io.wavfile import read
import tensorflow as tf
from tensorflow.nn.rnn_cell import LSTMCell
from tensorflow.nn import static_rnn


Fs = 44100
onset = 0.5
offset = 2.5
num_pitches = 88
min_pitch = 21
window_size = 2048
num_windows = int((offset-onset)*Fs/window_size)
num_hidden = 512
sess = tf.Session()


weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_pitches]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_pitches]))
}
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)


def preprocess():
    os.chdir("MAPS_data\\MAPS_AkPnBcht_1\\AkPnBcht\\ISOL\\NO")
    files = glob.glob("*.wav")
    filelength = int(Fs*(offset-onset))
    features = np.zeros((len(files)*num_windows, window_size))
    targets = np.zeros((len(files)*num_windows, num_pitches))
    for i in range(len(files)):
        file = np.mean(read(files[i])[1][int(Fs*onset):int(Fs*offset)], axis=1)
        pitch = int(files[i].split("_")[5][1:]) - min_pitch
        for j in range(num_windows):
            features[num_windows*i + j] = file[window_size*j:window_size*(j+1)]
            targets[num_windows*i + j][pitch] = 1
    for i in range(0, 5):
        os.chdir("..")
    shuffle_in_unison(features, targets)
    return features, targets


def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def RNN_predictor(x, weights, biases):
    x = tf.reshape(x, [-1, window_size])
    x = tf.split(x, window_size, 1)
    rnn_cell = LSTMCell(num_hidden, name='basic_lstm_cell')
    outputs, _ = static_rnn(rnn_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


def initialize_variables(learning_rate):
    prediction = RNN_predictor(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)
    # accuracy, _ = tf.metrics.accuracy(labels = tf.argmax(y, 1), predictions = tf.argmax(prediction, 1))
    sess.run(tf.global_variables_initializer())
    return prediction, cost, optimizer


def train_stochastic_epoch(features, targets, prediction, cost, optimizer, start, iterations):
    print("Training...")
    for i in range(iterations):
        _, loss = sess.run([optimizer, cost], feed_dict={x: features[(start+i)%len(features)], y: targets[(start+i)%len(features)]})
        print("Iteration: " + str(i) + "; Loss: " + str(loss))


features, targets = preprocess()
prediction, cost, optimizer = initialize_variables(learning_rate = 0.01)
train_stochastic_epoch(features, targets, prediction, cost, optimizer, 0, 2048)
