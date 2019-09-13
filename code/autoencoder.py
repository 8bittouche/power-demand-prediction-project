
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()
tf.set_random_seed(777)  # for reproducibility



learning_rate = 0.001
past_load_num = 5* 24
x_dim = 5 + 5 * 24 + past_load_num
h_dim = 250
iterations = 10000
drop_out = 1.0

train_set = np.loadtxt('C:\exercise_data\\original_hours_4_train.csv', delimiter=',', usecols=range(12))
test_set = np.loadtxt('C:\exercise_data\\original_hours_4_test.csv', delimiter=',', usecols=range(12))
train_dataX = []
train_dataY = []
test_dataX = []
test_dataY = []

i = past_load_num
num = len(train_set)
while i <= num - 24:
    for t in range(24):
        if train_set[i + t][11] == 0:
            i += 24
            continue

    tempX = []
    tempY = []
    for n in range(24):
        for j in range(11):
            if j == 0:
                continue
            elif n == 0 and j <= 5:
                tempX.append(train_set[i + n][j])
            elif j >= 6:
                tempX.append(train_set[i - 24 + n][j])

    for k in range(past_load_num):
        tempX.append(train_set[i - past_load_num + k][11])

    train_dataX.append(tempX)

    for m in range(24):
        tempY.append(train_set[i + m][11])
    train_dataY.append(tempY)
    # print(i)
    i += 24

i = past_load_num
num = len(test_set)
while i <= num - 24:
    for t in range(24):
        if test_set[i + t][11] == 0:
            i += 24
            continue

    tempX = []
    tempY = []
    for n in range(24):
        for j in range(11):
            if j == 0:
                continue
            elif n == 0 and j <= 5:
                tempX.append(test_set[i + n][j])
            elif j >= 6:
                tempX.append(test_set[i - 24 + n][j])

    for k in range(past_load_num):
        tempX.append(test_set[i - past_load_num + k][11])

    test_dataX.append(tempX)

    for m in range(24):
        tempY.append(test_set[i + m][11])
    test_dataY.append(tempY)
    # print(i)
    i += 24

trainX = np.array(train_dataX)
trainY = np.array(train_dataY)
testX = np.array(test_dataX)
testY = np.array(test_dataY)

print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)

# Training Parameters
AE_learning_rate = 0.001
AE_num_steps = 10000

# Network Parameters
num_hidden_1 = 120  # 1st layer num features
num_hidden_2 = 60  # 2nd layer num features (the latent dim)
num_input = x_dim

X = tf.placeholder(tf.float32, [None, x_dim])
X2 = tf.placeholder(tf.float32, [None, x_dim])
Y = tf.placeholder(tf.float32, [None, 24])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2


# Construct model
encoder_op = encoder(X2)
decoder_op = decoder(encoder_op)

# Prediction
AE_y_pred = decoder_op
# Targets (Labels) are the input data.
AE_y_true = X2

# Define loss and optimizer, minimize the squared error
AE_loss = tf.reduce_mean(tf.pow(AE_y_true - AE_y_pred, 2))
AE_optimizer = tf.train.RMSPropOptimizer(AE_learning_rate).minimize(AE_loss)

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

W9 = tf.get_variable("W9", shape=[h_dim, 24], initializer=tf.contrib.layers.xavier_initializer())
b9 = tf.Variable(tf.random_normal([24]), name='bias9')
hypothesis = tf.sigmoid(tf.matmul(layer2, W9) + b9)

# cost/loss function
loss = tf.reduce_mean(tf.square(hypothesis - Y))

train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# MAPE
targets = tf.placeholder(tf.float32, [None, 24])
predictions = tf.placeholder(tf.float32, [None, 24])
mape = tf.reduce_mean(tf.abs((targets - predictions) / targets)) * 100

test_set_mape = []
train_set_mape = []
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1, AE_num_steps + 1):
        _, l = sess.run([AE_optimizer, AE_loss], feed_dict={X2: trainX})
        # Display logs per step
        print('Step %i: Loss: %f' % (i, l))

    AE_X = sess.run(AE_y_pred, feed_dict={X2: trainX})

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
            X: AE_X, Y: trainY})
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