
import tensorflow as tf
import numpy as np

tf.reset_default_graph()
tf.set_random_seed(777)  # for reproducibility

import matplotlib.pyplot as plt

learning_rate = 0.001
past_load_num = 14
x_dim = 8 + past_load_num
h_dim = 25
iterations = 3500
drop_out = 1.0

train_set = np.loadtxt('C:\exercise_data\\original_month_1212_train_.csv', delimiter=',', usecols=range(10))
test_set = np.loadtxt('C:\exercise_data\\original_month_1212_test_.csv', delimiter=',', usecols=range(10))
train_dataX = []
train_dataY = []
test_dataX = []
test_dataY = []

i = past_load_num
num = len(train_set)
while i < num:
    if train_set[i][9] == 0:
        i += 1
        continue

    tempX = []
    tempY = []
    for j in range(9):
        if j == 0:
            continue
        elif j < 4:
            tempX.append(train_set[i][j])
        else:
            tempX.append(train_set[i - 1][j])

    for k in range(past_load_num):
        tempX.append(train_set[i - past_load_num + k][9])

    train_dataX.append(tempX)
    tempY.append(train_set[i][9])
    train_dataY.append(tempY)
    print(i)
    i += 1

i = past_load_num
num = len(test_set)
while i < num:
    if test_set[i][9] == 0:
        i += 1
        continue

    tempX = []
    tempY = []
    for j in range(9):
        if j == 0:
            continue
        elif j < 4:
            tempX.append(test_set[i][j])
        else:
            tempX.append(test_set[i - 1][j])

    for k in range(past_load_num):
        tempX.append(test_set[i - past_load_num + k][9])

    test_dataX.append(tempX)
    tempY.append(test_set[i][9])
    test_dataY.append(tempY)
    print(i)
    i += 1

trainX = np.array(train_dataX)
trainY = np.array(train_dataY)
testX = np.array(test_dataX)
testY = np.array(test_dataY)

print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)

X = tf.placeholder(tf.float32, [None, x_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# relu
W1 = tf.get_variable("W1", shape=[x_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([h_dim]), name='bias1')
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
layer1 = tf.nn.dropout(layer1, keep_prob=drop_out)

W2 = tf.get_variable("W2", shape=[h_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([h_dim]), name='bias2')
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob=drop_out)

'''
W3 = tf.get_variable("W3", shape=[h_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([h_dim]), name='bias3')
layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)

W4 = tf.get_variable("W4", shape=[h_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([h_dim]), name='bias4')
layer4 = tf.nn.relu(tf.matmul(layer3, W4) + b4)


W5 = tf.get_variable("W5", shape=[h_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([h_dim]), name='bias5')
layer5 = tf.nn.relu(tf.matmul(layer4, W5) + b5)


W6 = tf.get_variable("W6", shape=[h_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([h_dim]), name='bias6')
layer6 = tf.nn.relu(tf.matmul(layer5, W6) + b6)

W7 = tf.get_variable("W7", shape=[h_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([h_dim]), name='bias7')
layer7 = tf.nn.relu(tf.matmul(layer6, W7) + b7)

W8 = tf.get_variable("W8", shape=[h_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
b8 = tf.Variable(tf.random_normal([h_dim]), name='bias8')
layer8 = tf.nn.relu(tf.matmul(layer7, W8) + b8)
'''

W9 = tf.get_variable("W9", shape=[h_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
b9 = tf.Variable(tf.random_normal([1]), name='bias9')
hypothesis = tf.sigmoid(tf.matmul(layer2, W9) + b9)

'''
#sigmoid
W1 = tf.get_variable("W1", shape=[x_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([h_dim]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.get_variable("W2", shape=[h_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([h_dim]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

W3 = tf.get_variable("W3", shape=[h_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([h_dim]), name='bias3')
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W4 = tf.get_variable("W4", shape=[h_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([h_dim]), name='bias4')
layer4 = tf.sigmoid(tf.matmul(layer3, W4) + b4)

W5 = tf.get_variable("W5", shape=[h_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([h_dim]), name='bias5')
layer5 = tf.sigmoid(tf.matmul(layer4, W5) + b5)

W6 = tf.get_variable("W6", shape=[h_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([h_dim]), name='bias6')
layer6 = tf.sigmoid(tf.matmul(layer5, W6) + b6)

W7 = tf.get_variable("W7", shape=[h_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([h_dim]), name='bias7')
layer7 = tf.sigmoid(tf.matmul(layer6, W7) + b7)

W8 = tf.get_variable("W8", shape=[h_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
b8 = tf.Variable(tf.random_normal([h_dim]), name='bias8')
layer8 = tf.sigmoid(tf.matmul(layer7, W8) + b8)

W9 = tf.get_variable("W9", shape=[h_dim, 1], initializer=tf.contrib.layers.xavier_initializer())
b9 = tf.Variable(tf.random_normal([1]), name='bias9')
hypothesis = tf.sigmoid(tf.matmul(layer8, W9) + b9)
'''

# cost/loss function
loss = tf.reduce_mean(tf.square(hypothesis - Y))

train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# MAPE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
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

        # test set mape
        test_predict = sess.run(hypothesis, feed_dict={X: testX})
        test_mape = sess.run(mape, feed_dict={
            targets: testY, predictions: test_predict})
        test_set_mape.append(test_mape)

        # train set mape
        train_predict = sess.run(hypothesis, feed_dict={X: trainX})
        train_mape = sess.run(mape, feed_dict={
            targets: trainY, predictions: train_predict})
        train_set_mape.append(train_mape)

    # test set Test step
    test_pred = sess.run(hypothesis, feed_dict={X: testX})
    mape_val = sess.run(mape, feed_dict={
        targets: testY, predictions: test_pred})
    print("test set MAPE: {}".format(mape_val))

    # traing set Test step
    train_pred = sess.run(hypothesis, feed_dict={X: trainX})
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