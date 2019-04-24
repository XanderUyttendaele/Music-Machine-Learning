import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
threshold = 0.5
output = None

song = np.load("E:\\musicdata\\Music-Machine-Learning\\song_data_training\\MAPS_MUS-alb_esp2_AkPnStgb.npy")
print("Song loaded.")
weights = np.load("E:\\musicdata\\Music-Machine-Learning\\weights.npy")
bias = np.load("E:\\musicdata\\Music-Machine-Learning\\bias.npy")
print("Weights and bias loaded.")
length = song.shape[0]
num_classes = 252
state_size = 64


def print_ops():
    graph = tf.get_default_graph()
    ops = graph.get_operations()
    print("Total ops:", len(ops))
    for op in ops:
        print(op.name, op.type)


# with tf.Graph().as_default():
#     cell = tf.contrib.rnn.LSTMCell(state_size)
#     batch_size = 1
#     x_input = tf.placeholder(tf.float32, [batch_size, length, num_classes], name='x_input')
#     cell_state = tf.placeholder(tf.float32, [batch_size, state_size], name='cell_state')
#     hidden_state = tf.placeholder(tf.float32, [batch_size, state_size], name='hidden_state')
#     initial_state = tf.nn.rnn_cell.LSTMStateTuple(hidden_state, cell_state)
#     # list of 1 element with tensor 1 x VECTOR_SIZE
#     rnn_inputs = tf.unstack(x_input, num=length, axis=1)
#     outputs, current_state = tf.nn.static_rnn(cell, rnn_inputs, initial_state, dtype=tf.float32)
#     # Add ops to save and restore all the variables.
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         # Restore variables including RNN weights
#         saver.restore(sess, "E:\\musicdata\\Music-Machine-Learning\\saved_models\\model.ckpt")
#         print("Model restored.")
#         print_ops()
#
#         x_count = batch_size * length * num_classes # = 4
#         output_array, _next_state = sess.run([outputs, current_state], feed_dict={x_input: np.expand_dims(song, 0), cell_state: np.load("E:\\musicdata\\Music-Machine-Learning\\output.npy"), hidden_state: np.load("E:\\musicdata\\Music-Machine-Learning\\hidden.npy")})
#         _current_cell_state, _current_hidden_state = _next_state
#         output_array = [tf.squeeze(x) for x in output_array]
#         y_pred = tf.matmul(output_array, tf.convert_to_tensor(weights)) + tf.convert_to_tensor(bias)
#         y_pred = tf.greater(tf.nn.sigmoid(y_pred), threshold)
#         print("Output after sigmoid:", y_pred)
#         output = tf.Session().run(y_pred)
sess = tf.Session()
saver = tf.train.import_meta_graph("E:\\musicdata\\Music-Machine-Learning\\saved_models\\model.ckpt.meta")
saver.restore(sess, tf.train.latest_checkpoint("E:\\musicdata\\Music-Machine-Learning\\saved_models\\"))
print("Model restored.")
graph = tf.get_default_graph()
x = graph.get_tensor_by_name("x:0")
xsize = graph.get_tensor_by_name("xsize:0")
feed_dict = {xsize:length,x:[song]}
op_to_restore = graph.get_tensor_by_name("fullPrediction:0")
output = sess.run(op_to_restore, feed_dict)
