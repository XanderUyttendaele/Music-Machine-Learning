import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
import os
import os.path as pt
import math


training_dir = "song_data\\song_data_training\\"
target_dir = "song_data\\song_data_labeled\\"
num_input = 252
n_classes = 88
trainingPortions = 8
validationPortions = 2
stepCount = int(math.floor(2/(512/22050.0)))  # 2 seconds, assuming 512 hop length and 22050 rate
keyRange = tf.convert_to_tensor(np.arange(n_classes))


def load_data():
    clipCount = 0
    for root, directories, filenames in os.walk(training_dir):
        for filename in filenames:
            file_path = pt.join(root, filename)
            temp = np.load(file_path)
            clipCount += int(math.floor(temp.shape[0]/stepCount))

    noteData = np.empty([clipCount, stepCount, num_input])
    noteTargets = np.empty([clipCount, n_classes])

    x = 0
    for root, directories, filenames in os.walk(training_dir):
        for filename in filenames:
            file_path = pt.join(root, filename)
            temp = np.load(file_path)
            clips = int(math.floor(temp.shape[0]/stepCount))
            for i in range(0, clips):
                noteData[x] = temp[stepCount*i:stepCount*(i+1)]
                x += 1

    x = 0
    for root, directories, filenames in os.walk(target_dir):
        for filename in filenames:
            file_path = pt.join(root, filename)
            temp = np.load(file_path)
            clips = int(math.floor(temp.shape[0]/stepCount))
            for i in range(0, clips):
                noteTargets[x] = temp[stepCount*(i+1)-1]
                x+=1
    portionSize = clipCount // 10

    # different pianos, do this to ensure we're not fitting to a single piano
    noteData, noteTargets = randomize(noteData, noteTargets)
    trainingData = noteData[:portionSize * trainingPortions]
    validationData = noteData[portionSize * trainingPortions:portionSize * (trainingPortions + validationPortions):]
    # testData = noteData[portionSize * (trainingPortions + validationPortions):]

    trainingTargets = noteTargets[:portionSize * trainingPortions]
    validationTargets = noteTargets[portionSize * trainingPortions:portionSize * (trainingPortions + validationPortions):]
    # testTargets = noteTargets[portionSize * (trainingPortions + validationPortions):]

    return trainingData, validationData, trainingTargets, validationTargets


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
    return tf.get_variable('W', dtype=tf.float32, shape=shape, initializer=initer)


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
    outputs, current_state = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights) + biases


# x is for data, y is for targets
x_train, x_valid, y_train, y_valid = load_data()

learning_rate = 0.01    # The optimization initial learning rate
epochs = 10             # Total number of training epochs
batch_size = 100        # Training batch size
display_freq = 100      # Frequency of displaying the training results
threshold = 0.5         # Threshold for determining a "note"
num_hidden = 64         # Number of hidden units of the RNN


def build_graph(learning_rate, num_hidden, threshold):
    # Placeholders for inputs (x) and outputs(y)
    x = tf.placeholder(tf.float32, shape=(None, stepCount, num_input))
    y = tf.placeholder(tf.float32, shape=(None, n_classes))

    # create weight matrix initialized randomly from N~(0, 0.01)
    W = weight_variable(shape=[num_hidden, n_classes])

    # create bias vector initialized as zero
    b = bias_variable(shape=[n_classes])

    output_logits = RNN(x, W, b, stepCount, num_hidden)
    y_pred = tf.nn.sigmoid(output_logits)

    # Define the loss function, optimizer, and accuracy
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)

    # Accuracy: the percentage of individual notes it gets right.
    prediction = tf.greater(y_pred, threshold)
    accuracy = tf.metrics.accuracy(y, prediction)[1]
    stream_vars_acc = [v for v in tf.local_variables() if 'accuracy/' in v.name]

    # Creating the ops for initializing all variables
    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()

    sess = tf.InteractiveSession()
    sess.run(init_g)
    sess.run(init_l)

    return sess, stream_vars_acc, loss, optimizer, prediction, accuracy, x, y, W, b


def train(x_train, y_train, sess, stream_vars_acc, loss, optimizer, accuracy):
    training_loss = []
    training_accuracies = []
    validation_loss = []
    validation_accuracies = []

    # Number of training iterations in each epoch
    num_tr_iter = int(y_train.shape[0] / batch_size)
    for epoch in range(1, epochs+1):
        print('Training epoch: {}'.format(epoch))
        x_train, y_train = randomize(x_train, y_train)
        loss_batch, acc_batch = [0, 0]
        for iteration in range(num_tr_iter):
            start = iteration * batch_size
            end = (iteration + 1) * batch_size
            x_batch, y_batch = get_next_batch(x_train, y_train, start, end)
            # Run optimization op (backprop)
            feed_dict_batch = {x: x_batch, y: y_batch}

            # Calculate and display the batch loss and accuracy
            _, loss_batch, acc_batch = sess.run([optimizer, loss, accuracy], feed_dict=feed_dict_batch)
        training_loss.append(loss_batch)  # Loss for just last batch
        training_accuracies.append(acc_batch)  # Averaged accuracy over epoch
        print("Training Epoch {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".format(epoch, loss_batch, acc_batch))

        # Reset accuracy op, so validation accuracy can be separate.
        sess.run(tf.variables_initializer(stream_vars_acc))

        # Run validation after every epoch
        feed_dict_valid = {x: x_valid, y: y_valid}
        loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
        print('---------------------------------------------------------')
        print("Validation Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
              format(epoch, loss_valid, acc_valid))
        print('---------------------------------------------------------')
        validation_loss.append(loss_valid)
        validation_accuracies.append(acc_valid)

        # Reset accuracy op (otherwise calculates cumulative accuracy, which we probably don't want).
        sess.run(tf.variables_initializer(stream_vars_acc))

    return training_loss, training_accuracies, validation_loss, validation_accuracies


def plot_results(losses, accuracies, train_or_val):
    epoch_range = np.arange(epochs)
    plt.figure(1)
    plt.plot(epoch_range, losses, '-', label=train_or_val + ' Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.grid(True)
    plt.legend(loc='lower left')
    plt.show()

    plt.figure(2)
    plt.plot(epoch_range, accuracies, '-', label=train_or_val + ' Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()


sess, stream_vars_acc, loss, optimizer, prediction, accuracy, x, y, W, b = build_graph(learning_rate, num_hidden, threshold)
training_loss, training_accuracies, validation_loss, validation_accuracies = train(x_train, y_train, sess, stream_vars_acc, loss, optimizer, accuracy)
plot_results(training_loss, training_accuracies, 'Training')
plot_results(validation_loss, validation_accuracies, 'Validation')
