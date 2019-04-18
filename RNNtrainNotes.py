import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
import random
import os
import os.path as pt

training_dir = "E:\\musicdata\\Music-Machine-Learning\\note_training_data\\"
target_dir = "E:\\musicdata\\Music-Machine-Learning\\note_labeled_data\\"
num_input = 252
timesteps = np.load(training_dir + random.choice(os.listdir(training_dir))).shape[0]
n_classes = 88
trainingPortions = 8
validationPortions = 2
testPortions = 0
testLoss = []
validLoss = []
testAcc = []
validAcc = []

def load_data():
    fileCount = len([name for name in os.listdir(training_dir) if os.path.isfile(os.path.join(training_dir, name))])
    print(fileCount)
    print(timesteps)
    print(num_input)
    noteData = np.empty([fileCount, timesteps, num_input])
    noteTargets = np.empty([fileCount, n_classes])
    
    x = 0
    for root, directories, filenames in os.walk(training_dir):
        for filename in filenames:
            file_path = pt.join(root, filename)
            noteData[x] = np.load(file_path)
            x+=1
    x = 0
    for root, directories, filenames in os.walk(target_dir):
        for filename in filenames:
            file_path = pt.join(root, filename)
            noteTargets[x] = np.load(file_path)
            x+=1
    portionSize = fileCount // 10
    # different pianos, do this to ensure we're not fitting to a single piano
    noteData, noteTargets = randomize(noteData, noteTargets)
    trainingData = noteData[:portionSize * trainingPortions]
    validationData = noteData[portionSize * trainingPortions:portionSize * (trainingPortions + validationPortions):]
    # testData = noteData[portionSize * (trainingPortions + validationPortions):]
    trainingTargets = noteTargets[:portionSize * trainingPortions]
    validationTargets = noteTargets[portionSize * trainingPortions:portionSize * (trainingPortions + validationPortions):]
    # testTargets = noteTargets[portionSize * (trainingPortions + validationPortions):]
    return trainingData, validationData, trainingTargets, validationTargets # removed test


def randomize(x, y):
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y

def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch

# weight and bias wrappers
def weight_variable(shape):
    """
    Create a weight variable with appropriate initialization
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W',
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)

def bias_variable(shape):
    """
    Create a bias variable with appropriate initialization
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b',
                           dtype=tf.float32,
                           initializer=initial)

def RNN(x, weights, biases, timesteps, num_hidden):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a rnn cell with tensorflow
    lstm_cell = rnn.LSTMCell(num_hidden)

    # Get lstm cell output
    # If no initial_state is provided, dtype must be specified
    # If no initial cell state is provided, they will be initialized to zero
    states_series, current_state = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    print(current_state)
    # Linear activation, using rnn inner loop last output
    print(current_state[0])
    return tf.matmul(current_state[1], weights) + biases

# x is for data, y is for targets
x_train, x_valid, y_train, y_valid = load_data() # removed test
print(x_train.shape)
print(x_valid.shape)
print(y_train.shape)
print(y_valid.shape)
print("Size of:")
print("- Training-set:\t\t{}".format(len(y_train)))
print("- Validation-set:\t{}".format(len(y_valid)))
# print("- Test-set\t{}".format(len(y_test)))

learning_rate = 0.01  # The optimization initial learning rate
epochs = 1000         # Total number of training epochs
batch_size = 100      # Training batch size
display_freq = 100    # Frequency of displaying the training results

num_hidden_units = 15  # Number of hidden units of the RNN

# Placeholders for inputs (x) and outputs(y)
x = tf.placeholder(tf.float32, shape=(None, timesteps, num_input))
y = tf.placeholder(tf.float32, shape=(None, n_classes)) 

# create weight matrix initialized randomly from N~(0, 0.01)
W = weight_variable(shape=[num_hidden_units, n_classes])

# create bias vector initialized as zero
b = bias_variable(shape=[n_classes])

output_logits = RNN(x, W, b, timesteps, num_hidden_units)
y_pred = tf.nn.softmax(output_logits)

# Model predictions
cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')

# Define the loss function, optimizer, and accuracy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=output_logits), axis = 0, name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)
correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# Creating the op for initializing all variables
init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)
global_step = 0
# Number of training iterations in each epoch
num_tr_iter = int(len(y_train) / batch_size)
for epoch in range(epochs):
    print('Training epoch: {}'.format(epoch + 1))
    x_train, y_train = randomize(x_train, y_train)
    for iteration in range(num_tr_iter):
        global_step += 1
        start = iteration * batch_size
        end = (iteration + 1) * batch_size
        x_batch, y_batch = get_next_batch(x_train, y_train, start, end)
        # Run optimization op (backprop)
        feed_dict_batch = {x: x_batch, y: y_batch}
        sess.run(optimizer, feed_dict=feed_dict_batch)

        if iteration % display_freq == 0:
            # Calculate and display the batch loss and accuracy
            loss_batch, acc_batch = sess.run([loss, accuracy],
                                             feed_dict=feed_dict_batch)

            print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
                  format(iteration, loss_batch, acc_batch))
            testLoss.append(loss_batch)
            testAcc.append(acc_batch)

    # Run validation after every epoch

    feed_dict_valid = {x: x_valid[:1000].reshape((-1, timesteps, num_input)), y: y_valid[:1000]}
    loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
    print('---------------------------------------------------------')
    print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
          format(epoch + 1, loss_valid, acc_valid))
    print('---------------------------------------------------------')
    validLoss.append(loss_valid)
    validAcc.append(acc_batch)
epochRange = np.arange(epochs)
plt.figure(1)
plt.plot(epochRange,testLoss,'-',label='Test Loss')
plt.plot(epochRange,validLoss,'-',label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs')
plt.grid(True)
plt.legend(loc='lower left')
plt.show()

plt.figure(2)
plt.plot(epochRange,testAcc,'-',label='Test Accuracy')
plt.plot(epochRange,validAcc,'-',label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epochs')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()