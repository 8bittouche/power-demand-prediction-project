import tensorflow as tf
import numpy as np
import matplotlib
import os

tf.reset_default_graph()
tf.set_random_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

# train Parameters
seq_length = 7
data_dim = 10
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 1000
layer_num = 2
dropout = 1.0

# Open train_data, test_data
train_set = np.loadtxt('C:\exercise_data\\original_month_1212_train_.csv', delimiter=',', usecols=range(data_dim))
test_set = np.loadtxt('C:\exercise_data\\original_month_1212_test_.csv', delimiter=',', usecols=range(data_dim))


# build datasets
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i + seq_length, :]
        _y = time_series[i + seq_length, [-1]]  # Next close price
        print(_x, "->", _y)
        # print(_y.shape)
        if (_y != 0):
            dataX.append(_x)
            dataY.append(_y)
    return np.array(dataX), np.array(dataY)


trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length)

print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)
print(len(trainX))
print(len(trainY))
print(len(testX))
print(len(testY))

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])


# build a LSTM network
def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
    return cell


# build a GRU network
def gru_cell():
    cell = tf.contrib.rnn.GRUCell(
        num_units=hidden_dim, activation=tf.tanh)
    return cell


# cell = tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=dropout)
# multi_cells = tf.contrib.rnn.MultiRNNCell([cell for _ in range(layer_num)], state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(lstm_cell(), X, dtype=tf.float32)

# print(type(outputs))
# print(outputs.shape)
# print(outputs[:, -1])

Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
# MAPE
mape = tf.reduce_mean(tf.abs((targets - predictions) / targets)) * 100

test_set_mape = []
train_set_mape = []
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
            X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

        # Test step
        test_predict = sess.run(Y_pred, feed_dict={X: testX})
        test_mape = sess.run(mape, feed_dict={
            targets: testY, predictions: test_predict})
        test_set_mape.append(test_mape)

        # Train step
        train_predict = sess.run(Y_pred, feed_dict={X: trainX})
        train_mape = sess.run(mape, feed_dict={
            targets: trainY, predictions: train_predict})
        train_set_mape.append(train_mape)

    # Test step
    test_pred = sess.run(Y_pred, feed_dict={X: testX})
    mape_val = sess.run(mape, feed_dict={
        targets: testY, predictions: test_pred})
    print("test set MAPE: {}".format(mape_val))

    # train set Test step
    train_pred = sess.run(Y_pred, feed_dict={X: trainX})
    mape_val2 = sess.run(mape, feed_dict={
        targets: trainY, predictions: train_pred})
    print("train set MAPE: {}".format(mape_val2))

# Plot predictions
plt.figure()
# plt.plot(iteration, label='iterations')
plt.plot(test_set_mape, label='test set')
plt.plot(train_set_mape, label='train set')
plt.xlabel("iterations")
plt.ylabel("MAPE(%)")
plt.legend()
plt.show()
plt.clf()

# Plot predictions
plt.figure()
plt.plot(testY, label='original')
plt.plot(test_predict, label='prediction')
plt.xlabel("Time Period")
plt.ylabel("Maximum Electric Power")
plt.legend()
plt.show()