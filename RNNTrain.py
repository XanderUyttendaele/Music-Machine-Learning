import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
import os
import os.path as pt
import math


# Initialize variables related to data loading.
training_dir = "song_data\\song_data_training\\"
target_dir = "song_data\\song_data_labeled\\"
num_input = 252
n_classes = 88
trainingPortions = 8
validationPortions = 2
stepCount = int(math.floor(0.5/(512/22050.0)))  # 0.5 seconds, assuming 512 hop length and 22050 rate
keyRange = tf.convert_to_tensor(np.arange(n_classes))


def load_data(size):
    """
    Load a number of files from training_dir and target_dir equal to size parameter,
    then splits data into training and validation sets according to training_portions
    global variable.
    :param size: number of training examples loaded
    :return: training labels, training targets, validation data, and validation targets
    """
    noteData = np.empty([size, stepCount, num_input])
    noteTargets = np.empty([size, n_classes])
    x = 0
    for root, directories, filenames in os.walk(training_dir):
        for filename in filenames:
            if x < size:
                file_path = pt.join(root, filename)
                temp = np.load(file_path)
                clips = int(math.floor(temp.shape[0]/stepCount))
                for i in range(0, clips):
                    if x < size:
                        noteData[x] = temp[stepCount*i:stepCount*(i+1)]
                        x += 1

    x = 0
    for root, directories, filenames in os.walk(target_dir):
        for filename in filenames:
            if x < size:
                file_path = pt.join(root, filename)
                temp = np.load(file_path)
                clips = int(math.floor(temp.shape[0]/stepCount))
                for i in range(0, clips):
                    if x < size:
                        noteTargets[x] = temp[stepCount*(i+1)-1]
                        x+=1
    portionSize = size / 10

    # Shuffle, then partition, our loaded data.
    noteData, noteTargets = randomize(noteData, noteTargets)
    trainingData = noteData[:int(portionSize * trainingPortions)]
    validationData = noteData[int(portionSize * trainingPortions):int(portionSize * (trainingPortions + validationPortions)):]
    trainingTargets = noteTargets[:int(portionSize * trainingPortions)]
    validationTargets = noteTargets[int(portionSize * trainingPortions):int(portionSize * (trainingPortions + validationPortions)):]

    return trainingData, validationData, trainingTargets, validationTargets


def randomize(x, y):
    """
    Shuffle two numpy arrays in unison (according to the same permutation).
    :param x: the first numpy array
    :param y: the second numpy array
    :return: shuffled versions of both arrays
    """
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y


def get_next_batch(x, y, start, end):
    """
    Slices two numpy arrays in unison according to the same start
    and end indices.
    :param x: the first numpy array
    :param y: the second numpy array
    :return: corresponding slices of arrays
    """
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch


def weight_variable(shape):
    """
    Create a weight variable with appropriate initialization
    :param shape: weight shape
    :return: initialized weight variable
    """
    init = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W', dtype=tf.float32, shape=shape, initializer=init)


def bias_variable(shape):
    """
    Create a bias variable with appropriate initialization
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    init = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b', dtype=tf.float32, initializer=init)


def RNN(x, weights, biases, timesteps, num_hidden):
    """
    Defines an long-short-term-memory cell according to the parameters.
    :param x: input to LSTM; a placeholder at time of initialization
    :param weights: a 2D array of weights of size (num_hidden, n_classes) to be
        multiplied with the output of the pure LSTM cell.
    :param biases: an 1D array of biases of length (n_classes) to be added to the
        product of the matrix multiplication with the output of the pure LSTM cell.
    :param timesteps: the number of timesteps the LSTM runs through before
        outputting its prediction
    :param num_hidden: the number of hidden layers of the LSTM (i.e. the number of nodes
        in the pure LSTM output.
    :return: the output of the LSTM operated on by weights and biases
    """

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
x_train, x_valid, y_train, y_valid = load_data(20000)

# Initialize model hyperparameters
learning_rate = 0.01    # The optimization initial learning rate
epochs = 50             # Total number of training epochs - change back later, testing
batch_size = 100        # Training batch size
threshold = 0.8         # Threshold for determining a "note"
num_hidden = 128        # Number of hidden units of the RNN


def build_graph(learning_rate, num_hidden, threshold):
    """
    Build the tensorflow graph that will be used for training
    :param learning_rate: the learning rate of the Adam optimizer
    :param num_hidden: the number of hidden units in the RNN
    :param threshold: the classification threshold for whether a given note is being played
    :return: all relevant nodes of the graph
    """

    sess = tf.InteractiveSession()
    # Placeholders for inputs (x) and outputs(y)
    x = tf.placeholder(tf.float32, shape=(None, None, num_input), name = "x")
    y = tf.placeholder(tf.float32, shape=(None, n_classes))

    # Create weight matrix initialized randomly from N~(0, 0.01)
    W = weight_variable(shape=[num_hidden, n_classes])

    # Create bias vector initialized as zero
    b = bias_variable(shape=[n_classes])
    output_logits = RNN(x, W, b, stepCount, num_hidden)

    y_pred = tf.nn.sigmoid(output_logits)

    # Define the loss function, optimizer, and accuracy
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op').minimize(loss)

    # Initialize metrics used for model evaluation
    prediction = tf.greater(y_pred, threshold, name = "prediction")
    accuracy = tf.metrics.accuracy(y, prediction)[1]
    precision = tf.metrics.precision(y, prediction)[1]
    recall = tf.metrics.recall(y, prediction)[1]
    stream_vars_acc = [v for v in tf.local_variables() if 'accuracy/' in v.name or 'precision/' in v.name or 'recall/' in v.name]

    # Creating the ops for initializing all variables
    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()

    # Initialize both global and local variables
    sess.run(init_g)
    sess.run(init_l)

    return sess, stream_vars_acc, loss, optimizer, prediction, accuracy, precision, recall, x, y, W, b #, output, fullPrediction


def train(batch_size, epochs, x_train, y_train, sess, stream_vars_acc, loss, optimizer, accuracy, precision, recall):
    """
    Train model according to above model hyperparameters and pre-initialized graph.
    :param batch_size: the number of training examples in each training batch
    :param epochs: the number of times run through all data in training
    :param x_train: training labels
    :param y_train: training targets
    :param sess: tensorflow interactive session being used to run graph
    :param stream_vars_acc: local variables related to accuracy metrics
    :param loss: loss function for model
    :param optimizer: optimizer as defined in build_graph()
    :param accuracy: accuracy metric for model
    :param precision: precision metric for model
    :param recall: recall metric for model
    :return: data of model's performance after each epoch
    """

    # Variables which will be used to store data on model performance
    training_loss = []
    training_accuracies = []
    training_precisions = []
    training_recalls = []
    validation_loss = []
    validation_accuracies = []
    validation_precisions = []
    validation_recalls = []
    saver = tf.train.Saver()

    # Number of training iterations in each epoch
    num_tr_iter = int(y_train.shape[0] / batch_size)

    # Train for epochs epochs.
    for epoch in range(1, epochs+1):
        print('Training epoch: {}'.format(epoch))
        print('---------------------------------------------------------')
        x_train, y_train = randomize(x_train, y_train)
        loss_batch, acc_batch, prec_batch, rec_batch = [0, 0, 0, 0]
        for iteration in range(num_tr_iter):
            start = iteration * batch_size
            end = (iteration + 1) * batch_size
            x_batch, y_batch = get_next_batch(x_train, y_train, start, end)
            feed_dict_batch = {x: x_batch, y: y_batch}
            # Calculate and display the batch loss and accuracy
            _, loss_batch, acc_batch, prec_batch, rec_batch = sess.run([optimizer, loss, accuracy, precision, recall], feed_dict=feed_dict_batch)
        training_loss.append(loss_batch)  # Loss for just last batch
        training_accuracies.append(acc_batch)  # Averaged accuracy over epoch
        training_precisions.append(prec_batch)
        training_recalls.append(rec_batch)
        print("Training Epoch {0:3d}: Loss={1:.2f}, Accuracy={2:.01%}, Precision={3:.01%}, Recall={4:.01%}".format(epoch, loss_batch, acc_batch, prec_batch, rec_batch))

        # Reset accuracy op, so validation accuracy can be separate.
        sess.run(tf.variables_initializer(stream_vars_acc))

        # Run validation after every epoch
        feed_dict_valid = {x: x_valid, y: y_valid}
        loss_valid, acc_valid, prec_valid, rec_valid = sess.run([loss, accuracy, precision, recall], feed_dict=feed_dict_valid)
        print('---------------------------------------------------------')
        print("Validation Epoch: {0}, Loss: {1:.2f}, Accuracy: {2:.01%}, Precision={3:.01%}, Recall={4:.01%}".
              format(epoch, loss_valid, acc_valid, prec_valid, rec_valid))
        print('---------------------------------------------------------')
        validation_loss.append(loss_valid)
        validation_accuracies.append(acc_valid)
        validation_precisions.append(prec_valid)
        validation_recalls.append(rec_valid)

        # Reset accuracy op (otherwise calculates cumulative accuracy, which we probably don't want).
        sess.run(tf.variables_initializer(stream_vars_acc))
        if epoch % 10 == 0:
            saver.save(sess, "saved_models\\test.ckpt")
            print("Model saved.")
    return training_loss, training_accuracies, training_precisions, training_recalls, validation_loss, \
           validation_accuracies, validation_precisions, validation_recalls


def plot_results(losses, accuracies, precisions, recalls, train_or_val):
    """
    Plot results of model (loss, accuracy, precision, recall vs epochs)
    :param losses: list of losses for each epoch of training
    :param accuracies: list of accuracies for each epoch of training
    :param accuracies: list of precisions for each epoch of training
    :param accuracies: list of recalls for each epoch of training
    :param train_or_val: whether this is data for the training set of the validation set
    """
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
    plt.title('Accuracy vs Epochs')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()

    plt.figure(3)
    plt.plot(epoch_range, precisions, '-', label=train_or_val + ' Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Precision vs Epochs')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()

    plt.figure(4)
    plt.plot(epoch_range, recalls, '-', label=train_or_val + ' Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Recall vs Epochs')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()

# Train model
sess, stream_vars_acc, loss, optimizer, prediction, accuracy, precision, recall, x, y, W, b= build_graph(learning_rate, num_hidden, threshold) #, output, fullPrediction = build_graph(learning_rate, num_hidden, threshold)
training_loss, training_accuracies, training_precisions, training_recalls, validation_loss, validation_accuracies, \
    validation_precisions, validation_recalls = train(batch_size, epochs, x_train, y_train, sess, stream_vars_acc, loss,
                                                      optimizer, accuracy, precision, recall)

# Plot results
plot_results(training_loss, training_accuracies, training_precisions, training_recalls, 'Training')
plot_results(validation_loss, validation_accuracies, validation_precisions, validation_recalls, 'Validation')
