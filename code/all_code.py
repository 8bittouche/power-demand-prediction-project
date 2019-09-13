# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import tensorflow as tf
import matplotlib.pyplot as plt





#%%
import tensorflow as tf
import numpy as np
import matplotlib
import os

tf.set_random_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


# train Parameters
seq_length = 7
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 500

# Open, High, Low, Volume, Close
xy = np.loadtxt('C:\exercise_data\data-02-stock_daily.csv', delimiter=',', usecols=range(5))
xy = xy[::-1]

print(xy.shape) #(732, 5)

# train/test split
train_size = int(len(xy) * 0.7)
train_set = xy[0:train_size]
test_set = xy[train_size - seq_length:]  # Index from [train_size - seq_length] to utilize past sequence

# Scale each
train_set = MinMaxScaler(train_set)
test_set = MinMaxScaler(test_set)

# build datasets
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i + seq_length, :]
        _y = time_series[i + seq_length, [-1]]  # Next close price
        print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)


trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length)

print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)

'''
# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
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

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
        #print("[step: {}] loss: {}".format(i, step_loss))

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={
                    targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))

    # Plot predictions
    plt.plot(testY)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()
'''





#%%
import tensorflow as tf
import numpy as np
import matplotlib
    
xy = np.loadtxt('C:\exercise_data\최대전력수급_test2.csv', delimiter=',', usecols=range(1))
xy = xy[::-1]
print(xy)
np.savetxt('C:\exercise_data\최대전력수급_test_수정3.csv', xy, delimiter=',')

f = open('C:\exercise_data\최대전력수급_test2.csv', 'r')
f2 = open('C:\exercise_data\최대전력수급_test_수정2.csv', 'w')
while True:
    line = f.readline()
    if not line: break
    data = line[:-1] + "," + "\n"        
    f2.write(data)
f.close()


    
    
    
    
#%%
import tensorflow as tf
import numpy as np
import matplotlib
import os

tf.set_random_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


# train Parameters
seq_length = 7
data_dim = 1
hidden_dim = 4
output_dim = 1
learning_rate = 0.01
iterations = 20000
layer_num = 2

# Open, High, Low, Volume, Close
xy = np.loadtxt('C:\exercise_data\최대전력수급_온도_test.csv', delimiter=',', usecols=range(2))
#xy = xy[::-1]
#print(xy)
#print(xy.shape) #(3179,)
#np.reshape(xy, (len(xy),1))
#print(xy.shape)
#print(xy)


# train/test split
train_size = int(len(xy) * 0.9)
train_set = xy[0:train_size]
test_set = xy[train_size - seq_length:]  # Index from [train_size - seq_length] to utilize past sequence

# Scale each
train_set = MinMaxScaler(train_set)
test_set = MinMaxScaler(test_set)

#print(len(xy)) #3179
#print(len(train_set)) #2543
#print(len(test_set)) #643


# build datasets
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i + seq_length]
        temp_x = []
        temp_y = []
        for j in _x:
            temp = []
            temp.append(j)
            temp_x.append(temp)            
        _y = time_series[i + seq_length]  # Next close price
        temp_y.append(_y)
        print(temp_x, "->", temp_y)
        dataX.append(temp_x)
        dataY.append(temp_y)
    return np.array(dataX), np.array(dataY)


trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length)

print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)


# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
    return cell

multi_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)], state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)

#print(type(outputs))
#print(outputs.shape)


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
    rmse_val = sess.run(rmse, feed_dict={
                    targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))

    # Plot predictions
    plt.figure()
    plt.plot(testY, label='original')
    plt.plot(test_predict, label='prediction')
    plt.xlabel("Time Period")
    plt.ylabel("Maximum Electric Power")
    plt.legend()
    plt.show()



#%%
import tensorflow as tf
import numpy as np
import matplotlib
import os

tf.set_random_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


# train Parameters
seq_length = 7
data_dim = 2
hidden_dim = 4
output_dim = 1
learning_rate = 0.01
iterations = 20000
layer_num = 2

# Open, High, Low, Volume, Close
xy = np.loadtxt('C:\exercise_data\최대전력수급_온도_test.csv', delimiter=',', usecols=range(2))
#xy = xy[::-1]
#print(xy)
#print(xy.shape) #(3179,)
#np.reshape(xy, (len(xy),1))
#print(xy.shape)
#print(xy)


# train/test split
train_size = int(len(xy) * 0.9)
train_set = xy[0:train_size]
test_set = xy[train_size - seq_length:]  # Index from [train_size - seq_length] to utilize past sequence

# Scale each
train_set = MinMaxScaler(train_set)
test_set = MinMaxScaler(test_set)

#print(len(xy)) #3179
#print(len(train_set)) #2543
#print(len(test_set)) #643


# build datasets
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i + seq_length, :]
        _y = time_series[i + seq_length, [-1]]  # Next close price
        print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)


trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length)

print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)


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

multi_cells = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(layer_num)], state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)

#print(type(outputs))
#print(outputs.shape)


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
    rmse_val = sess.run(rmse, feed_dict={
                    targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))

    # Plot predictions
    plt.figure()
    plt.plot(testY, label='original')
    plt.plot(test_predict, label='prediction')
    plt.xlabel("Time Period")
    plt.ylabel("Maximum Electric Power")
    plt.legend()
    plt.show()
    
    

#%%
import tensorflow as tf
import numpy as np
import matplotlib
import os
from sklearn import preprocessing

tf.set_random_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


# train Parameters
seq_length = 7
data_dim = 2
hidden_dim = 2
output_dim = 1
learning_rate = 0.01
iterations = 50000
layer_num = 2

# Open, High, Low, Volume, Close
xy = np.loadtxt('C:\exercise_data\\train_normalization_.csv', delimiter=',', usecols=range(8))
#xy = xy[::-1]
#print(xy)
#print(xy.shape) #(3179,)
#np.reshape(xy, (len(xy),1))
#print(xy.shape)
#print(xy)

'''
minValue = np.min(xy, 0)
maxValue = np.max(xy, 0)
numerator = xy - np.min(xy, 0)
print(xy)
print(numerator)
print(minValue)
print(maxValue)
'''






'''
# train/test split
train_size = int(len(xy) * 0.9)
train_set = xy[0:train_size]
test_set = xy[train_size - seq_length:]  # Index from [train_size - seq_length] to utilize past sequence

# Scale each
train_set = MinMaxScaler(train_set)
test_set = MinMaxScaler(test_set)

#print(len(xy)) #3179
#print(len(train_set)) #2543
#print(len(test_set)) #643


# build datasets
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i + seq_length, :]
        _y = time_series[i + seq_length, [-1]]  # Next close price
        print(_x, "->", _y)
        #print(_y.shape)
        if(_y!=0) :
            dataX.append(_x)
            dataY.append(_y)
    return np.array(dataX), np.array(dataY)


trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length)

print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)


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

multi_cells = tf.contrib.rnn.MultiRNNCell([gru_cell() for _ in range(layer_num)], state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)

#print(type(outputs))
#print(outputs.shape)


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
mape = tf.reduce_mean( tf.abs((targets - predictions)/targets) ) * 100

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
    mape_val = sess.run(mape, feed_dict={
                    targets: testY, predictions: test_predict})
    print("MAPE: {}".format(mape_val))

    # Plot predictions
    plt.figure()
    plt.plot(testY, label='original')
    plt.plot(test_predict, label='prediction')
    plt.xlabel("Time Period")
    plt.ylabel("Maximum Electric Power")
    plt.legend()
    plt.show()
'''

#%%
import tensorflow as tf
import numpy as np
import matplotlib
from sklearn import preprocessing
import math


xy = np.loadtxt('C:\exercise_data\\data_real_last3.csv', delimiter=',', usecols=range(8))

xy_scaled = preprocessing.scale(xy)
print(xy_scaled)
print(xy_scaled.mean(axis=0))
print(xy_scaled.std(axis=0))
print(len(xy_scaled))

np.savetxt('C:\exercise_data\\data_real_last4.csv', xy_scaled, delimiter=',')


'''
xy = np.loadtxt('C:\exercise_data\\train_normalization_.csv', delimiter=',', usecols=range(8))
xy = xy[::-1]
#print(xy)
np.savetxt('C:\exercise_data\최대전력수급_20060101_20180930_last6.csv', xy, delimiter=',')
'''


f = open('C:\exercise_data\date_20060101_20180930___test.csv', 'r')
f2 = open('C:\exercise_data\date_20060101_20180930___test3.csv', 'w')
while True:
    #line = f.read().split(',')
    line =  f.readline()
    if not line: break
    data = line.split(',')    
    #print(line.shape)    
    #print(lines)    
    #data = line[:]
    #print(data[0])
    #print(data[1])
    #print(data[2])
    #print(data[3])
    #print(data[4])
        
    if data[3]=='토':
        print(data[3])
        data[4] = 1
    elif data[3]=='일':
        data[4] = 1
        print(data[3])
    
    newline = str(data[0]) + ',' + str(data[1]) + ',' + str(data[2]) + ',' + str(data[3]) + ',' + str(data[4]) + ',' + '\n'
    print(newline)
    f2.write(newline)
    
f.close()


f = open('C:\exercise_data\\data_real_last.csv', 'r')
f2 = open('C:\exercise_data\\data_real_last2.csv', 'w')
while True:
    line = f.readline()
    if not line: break
    #data = line.split(',')
    #print(data)
    
    if data[5] == '':
        print(data[0]+'년'+data[1]+'월'+data[2]+'일')
    if data[6] == '':
        print(data[0]+'년'+data[1]+'월'+data[2]+'일')
    if data[7] == '':
        print(data[0]+'년'+data[1]+'월'+data[2]+'일')
    if data[8] == '':
        print(data[0]+'년'+data[1]+'월'+data[2]+'일')
    if data[9] == '':
        print(data[0]+'년'+data[1]+'월'+data[2]+'일')
    if data[10] == '':
        print(data[0]+'년'+data[1]+'월'+data[2]+'일')
    if data[11] == '':
        print(data[0]+'년'+data[1]+'월'+data[2]+'일')
    if data[12] == '':
        print(data[0]+'년'+data[1]+'월'+data[2]+'일')
    
    data = line[:-1] + "," + "\n"        
    f2.write(data)
f.close()


f = open('C:\exercise_data\\original_hours_month_345_test.csv', 'r')
f2 = open('C:\exercise_data\\original_hours_month_345_test_.csv', 'w')
while True:
    line = f.readline()
    if not line: break
  
    data = line[:-1] + "," + "\n"        
    f2.write(data)
f.close()


data = []
pi = 3.14159265359
xy = np.loadtxt('C:\exercise_data\\month_day_dayofweek3.csv', delimiter=',', usecols=range(3))
for i in range(len(xy)):
    temp = []
    #print(xy[i][1])
    if i < len(xy)-30:
        if xy[i][1] == 1:
            if xy[i+27][1] == 1:
                month_len = 28
            elif xy[i+28][1] == 1:
                month_len = 29
            elif xy[i+29][1] == 1:
                month_len = 30
            elif xy[i+30][1] == 1:
                month_len = 31
    else:
        month_len=30
    print(month_len)
    
    month = (0.25*math.sin((2*pi/12)*xy[i][0])+0.25) + (0.25*math.cos((2*pi/12)*xy[i][0]) + 0.25)
    day = (0.25*math.sin((2*pi/month_len)*xy[i][1])+0.25) + (0.25*math.cos((2*pi/month_len)*xy[i][1]) + 0.25)
    dayOfweek = (0.25*math.sin((2*pi/7)*xy[i][2])+0.25) + (0.25*math.cos((2*pi/7)*xy[i][2]) + 0.25)
    
    temp.append(month)
    temp.append(day)
    temp.append(dayOfweek)
    data.append(temp)

np.savetxt('C:\exercise_data\month_day_dayofweek_scaled.csv', np.array(data), delimiter=',')



xy = np.loadtxt('C:\exercise_data\\전력수요데이터\\20180701_20180930_.csv', delimiter=',', usecols=range(8))

data = []
for i in range(len(xy)):
    temp = []
    s = str(xy[i][0])
    if s[10]=='0' and s[11]=='0' and s[12]=='0' and s[13]=='0':
        temp.append(xy[i][0])
        temp.append(xy[i][1])
        temp.append(xy[i][2])
        temp.append(xy[i][3])
        temp.append(xy[i][4])
        temp.append(xy[i][5])
        temp.append(xy[i][6])
        temp.append(xy[i][7])
        data.append(temp)

np.savetxt('C:\exercise_data\\전력수요데이터\\20180701_20180930_hours.csv', np.array(data), delimiter=',')



gwangju = np.loadtxt('C:\exercise_data\\기상데이터\\광주_2016010100_2018093023_.csv', delimiter=',', usecols=range(12))
daegu = np.loadtxt('C:\exercise_data\\기상데이터\\대구_2016010100_2018093023_.csv', delimiter=',', usecols=range(12))
daejeon = np.loadtxt('C:\exercise_data\\기상데이터\\대전_2016010100_2018093023_.csv', delimiter=',', usecols=range(12))
busan = np.loadtxt('C:\exercise_data\\기상데이터\\부산_2016010100_2018093023_.csv', delimiter=',', usecols=range(12))
seoul = np.loadtxt('C:\exercise_data\\기상데이터\\서울_2016010100_2018093023_.csv', delimiter=',', usecols=range(12))

data = []
pi = 3.14159265359
for i in range(len(gwangju)):
    temp = []
    
    x1 = 0.08*gwangju[i][7] + 0.12*daegu[i][7] + 0.1*daejeon[i][7] + 0.2*busan[i][7] + 0.5*seoul[i][7]
    x2 = 0.08*gwangju[i][8] + 0.12*daegu[i][8] + 0.1*daejeon[i][8] + 0.2*busan[i][8] + 0.5*seoul[i][8]
    x3 = 0.08*gwangju[i][9] + 0.12*daegu[i][9] + 0.1*daejeon[i][9] + 0.2*busan[i][9] + 0.5*seoul[i][9]
    x4 = 0.08*gwangju[i][10] + 0.12*daegu[i][10] + 0.1*daejeon[i][10] + 0.2*busan[i][10] + 0.5*seoul[i][10]
    x5 = 0.08*gwangju[i][11] + 0.12*daegu[i][11] + 0.1*daejeon[i][11] + 0.2*busan[i][11] + 0.5*seoul[i][11]
    
    
    if i < len(gwangju)-30*24:
        if gwangju[i][3] == 1 and gwangju[i][4] == 0:
            if gwangju[i+28*24][3] == 1:
                month_len = 28
            elif gwangju[i+29*24][3] == 1:
                month_len = 29
            elif gwangju[i+30*24][3] == 1:
                month_len = 30
            elif gwangju[i+31*24][3] == 1:
                month_len = 31
    else:
        month_len=30
   
   
    year = gwangju[i][1]
    month = (0.25*math.sin((2*pi/12)*gwangju[i][2])+0.25) + (0.25*math.cos((2*pi/12)*gwangju[i][2]) + 0.25)
    day = (0.25*math.sin((2*pi/month_len)*gwangju[i][3])+0.25) + (0.25*math.cos((2*pi/month_len)*gwangju[i][3]) + 0.25)
    hours = (0.25*math.sin((2*pi/24)*gwangju[i][4])+0.25) + (0.25*math.cos((2*pi/24)*gwangju[i][4]) + 0.25)
    dayOfweek = (0.25*math.sin((2*pi/7)*gwangju[i][5])+0.25) + (0.25*math.cos((2*pi/7)*gwangju[i][5]) + 0.25)
    holiday = gwangju[i][6]
    
    temp.append(year)
    temp.append(month)
    temp.append(day)
    temp.append(hours)
    temp.append(dayOfweek)
    temp.append(holiday)
    temp.append(x1)
    temp.append(x2)
    temp.append(x3)
    temp.append(x4)
    temp.append(x5)
    
    data.append(temp)
    

np.savetxt('C:\exercise_data\\기상데이터\\sum_five_region.csv', np.array(data), delimiter=',')  



Monday = []
Tuesday = []
Wendsday = []
Thursday = []
Friday = []
Saturday = []
Sunday = []

xy = np.loadtxt('C:\exercise_data\\orginal_hours_4.csv', delimiter=',', usecols=range(18))
for i in range(len(xy)):
    temp = []
    
    temp.append(xy[i][0])
    temp.append(xy[i][1])
    temp.append(xy[i][2])
    temp.append(xy[i][3])
    temp.append(xy[i][5])
    temp.append(xy[i][6])
    temp.append(xy[i][7])
    temp.append(xy[i][8])
    temp.append(xy[i][9])
    temp.append(xy[i][10])
    temp.append(xy[i][12])
    
    if xy[i][4] == 1:
        Monday.append(temp)
    elif xy[i][4] == 2:
        Tuesday.append(temp)
    elif xy[i][4] == 3:
        Wendsday.append(temp)
    elif xy[i][4] == 4:
        Thursday.append(temp)
    elif xy[i][4] == 5:
        Friday.append(temp)
    elif xy[i][4] == 6:
        Saturday.append(temp)
    elif xy[i][4] == 7:
        Sunday.append(temp)

np.savetxt('C:\exercise_data\\Monday_hours.csv', np.array(Monday), delimiter=',')        
np.savetxt('C:\exercise_data\\Tuesday_hours.csv', np.array(Tuesday), delimiter=',')
np.savetxt('C:\exercise_data\\Wendsday_hours.csv', np.array(Wendsday), delimiter=',')
np.savetxt('C:\exercise_data\\Thursday_hours.csv', np.array(Thursday), delimiter=',')
np.savetxt('C:\exercise_data\\Friday_hours.csv', np.array(Friday), delimiter=',')
np.savetxt('C:\exercise_data\\Saturday_hours.csv', np.array(Saturday), delimiter=',')
np.savetxt('C:\exercise_data\\Sunday_hours.csv', np.array(Sunday), delimiter=',')

        

    

#%%
from sklearn import preprocessing


#%%
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


#Open train_data, test_data
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
        #print(_y.shape)
        if(_y!=0) :
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


#cell = tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=dropout)
#multi_cells = tf.contrib.rnn.MultiRNNCell([cell for _ in range(layer_num)], state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(lstm_cell(), X, dtype=tf.float32)

#print(type(outputs))
#print(outputs.shape)
#print(outputs[:, -1])

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
mape = tf.reduce_mean( tf.abs((targets - predictions)/targets) ) * 100

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
#plt.plot(iteration, label='iterations')
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


#%%    
import tensorflow as tf
import numpy as np    

    
xy = np.loadtxt('C:\exercise_data\\temp.csv', delimiter=',', usecols=range(7))    

def MinMaxScaler(data):
    
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


xy_scaled = MinMaxScaler(xy)
np.savetxt('C:\exercise_data\\temp_.csv', xy_scaled, delimiter=',')

#%%
import tensorflow as tf
import numpy as np

tf.reset_default_graph()
tf.set_random_seed(777)  # for reproducibility



learning_rate = 0.01
past_load_num = 28
x_dim = 7+past_load_num
h_dim = 40
iterations = 100000
drop_out = 0.5

train_set = np.loadtxt('C:\exercise_data\\train_value_min_max_scalar.csv', delimiter=',', usecols=range(8))
test_set = np.loadtxt('C:\exercise_data\\test_value_min_max_value.csv', delimiter=',', usecols=range(8))
train_dataX = []
train_dataY = []
test_dataX = []
test_dataY = []


i = past_load_num
num = len(train_set)
while i < num:
    if train_set[i][7]==0:
        i+=1
        continue
    
    tempX = []
    tempY = []
    for j in range(7):
        tempX.append(train_set[i][j])
    
    for k in range(past_load_num):
        tempX.append(train_set[i-past_load_num+k][7])
    
    train_dataX.append(tempX)
    tempY.append(train_set[i][7])
    train_dataY.append(tempY)
    print(i)
    i+=1

i = past_load_num
num = len(test_set)
while i < num:
    if test_set[i][7]==0:
        i+=1
        continue
    
    tempX = []
    tempY = []
    for j in range(7):
        tempX.append(test_set[i][j])
    
    for k in range(past_load_num):
        tempX.append(test_set[i-past_load_num+k][7])
    
    test_dataX.append(tempX)
    tempY.append(test_set[i][7])
    test_dataY.append(tempY)
    print(i)
    i+=1
    
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


#relu
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
mape = tf.reduce_mean( tf.abs((targets - predictions)/targets) ) * 100


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

    # test set Test step
    test_predict = sess.run(hypothesis, feed_dict={X: testX})
    mape_val = sess.run(mape, feed_dict={
                    targets: testY, predictions: test_predict})
    print("test set MAPE: {}".format(mape_val))
    
    # traing set Test step
    test_predict = sess.run(hypothesis, feed_dict={X: trainX})
    mape_val2 = sess.run(mape, feed_dict={
                    targets: trainY, predictions: test_predict})
    print("train set MAPE: {}".format(mape_val2))



#%%
import tensorflow as tf
import numpy as np

tf.reset_default_graph()
tf.set_random_seed(777)  # for reproducibility



learning_rate = 0.01
past_load_num = 28
x_dim = 10+past_load_num
h_dim = 40
iterations = 20000
drop_out = 0.7

train_set = np.loadtxt('C:\exercise_data\\train_value_min_max_scalar2.csv', delimiter=',', usecols=range(11))
test_set = np.loadtxt('C:\exercise_data\\test_value_min_max_value2.csv', delimiter=',', usecols=range(11))
train_dataX = []
train_dataY = []
test_dataX = []
test_dataY = []


i = past_load_num
num = len(train_set)
while i < num:
    if train_set[i][10]==0:
        i+=1
        continue
    
    tempX = []
    tempY = []
    for j in range(10):
        tempX.append(train_set[i][j])
    
    for k in range(past_load_num):
        tempX.append(train_set[i-past_load_num+k][10])
    
    train_dataX.append(tempX)
    tempY.append(train_set[i][10])
    train_dataY.append(tempY)
    print(i)
    i+=1

i = past_load_num
num = len(test_set)
while i < num:
    if test_set[i][10]==0:
        i+=1
        continue
    
    tempX = []
    tempY = []
    for j in range(10):
        tempX.append(test_set[i][j])
    
    for k in range(past_load_num):
        tempX.append(test_set[i-past_load_num+k][10])
    
    test_dataX.append(tempX)
    tempY.append(test_set[i][10])
    test_dataY.append(tempY)
    print(i)
    i+=1
    
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


#relu
W1 = tf.get_variable("W1", shape=[x_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([h_dim]), name='bias1')
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
layer1 = tf.nn.dropout(layer1, keep_prob=drop_out)

W2 = tf.get_variable("W2", shape=[h_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([h_dim]), name='bias2')
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob=drop_out)


W3 = tf.get_variable("W3", shape=[h_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([h_dim]), name='bias3')
layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)
'''
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
hypothesis = tf.sigmoid(tf.matmul(layer3, W9) + b9)


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
mape = tf.reduce_mean( tf.abs((targets - predictions)/targets) ) * 100


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

    # test set Test step
    test_predict = sess.run(hypothesis, feed_dict={X: testX})
    mape_val = sess.run(mape, feed_dict={
                    targets: testY, predictions: test_predict})
    print("test set MAPE: {}".format(mape_val))
    
    # traing set Test step
    test_predict = sess.run(hypothesis, feed_dict={X: trainX})
    mape_val2 = sess.run(mape, feed_dict={
                    targets: trainY, predictions: test_predict})
    print("train set MAPE: {}".format(mape_val2))


#%%
import tensorflow as tf
import numpy as np

tf.reset_default_graph()
tf.set_random_seed(777)  # for reproducibility

import matplotlib.pyplot as plt

learning_rate = 0.001
past_load_num = 14
x_dim = 8+past_load_num
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
    if train_set[i][9]==0:
        i+=1
        continue
    
    tempX = []
    tempY = []
    for j in range(9):
        if j==0:
            continue
        elif j<4:
            tempX.append(train_set[i][j])
        else:
            tempX.append(train_set[i-1][j])
    
    for k in range(past_load_num):
        tempX.append(train_set[i-past_load_num+k][9])
    
    train_dataX.append(tempX)
    tempY.append(train_set[i][9])
    train_dataY.append(tempY)
    print(i)
    i+=1

i = past_load_num
num = len(test_set)
while i < num:
    if test_set[i][9]==0:
        i+=1
        continue
    
    tempX = []
    tempY = []
    for j in range(9):
        if j==0:
            continue
        elif j<4:
            tempX.append(test_set[i][j])
        else:
            tempX.append(test_set[i-1][j])
    
    for k in range(past_load_num):
        tempX.append(test_set[i-past_load_num+k][9])
    
    test_dataX.append(tempX)
    tempY.append(test_set[i][9])
    test_dataY.append(tempY)
    print(i)
    i+=1
    
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


#relu
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
mape = tf.reduce_mean( tf.abs((targets - predictions)/targets) ) * 100


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
#plt.plot(iteration, label='iterations')
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


#%%    
   
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()
tf.set_random_seed(777)  # for reproducibility



learning_rate = 0.001
past_load_num = 1*7
x_dim = 9*24+past_load_num
h_dim = 250
iterations = 5000
drop_out = 1.0


train_set = np.loadtxt('C:\exercise_data\\요일별\\Monday_hours_train.csv', delimiter=',', usecols=range(11))
test_set = np.loadtxt('C:\exercise_data\\요일별\\Monday_hours_test.csv', delimiter=',', usecols=range(11))
train_dataX = []
train_dataY = []
test_dataX = []
test_dataY = []


i = past_load_num
num = len(train_set)
while i <= num-24:
    #if train_set[i][10]==0:
    #    i+=1
    #    continue
    
    tempX = []
    tempY = []
    for n in range(24):
        for j in range(10):
            if j==0:
                continue
            tempX.append(train_set[i-24+n][j])
       
    
    for k in range(past_load_num):
        tempX.append(train_set[i-past_load_num+k][10])
    
    train_dataX.append(tempX)
    
    for m in range(24):
        tempY.append(train_set[i+m][10])
    train_dataY.append(tempY)
    #print(i)
    i+=24

i = past_load_num
num = len(test_set)
while i <= num-24:
    #if train_set[i][10]==0:
    #    i+=1
    #    continue
    
    tempX = []
    tempY = []
    for n in range(24):
        for j in range(10):
            if j==0:
                continue
            tempX.append(test_set[i-24+n][j])
       
    
    for k in range(past_load_num):
        tempX.append(test_set[i-past_load_num+k][10])
    
    test_dataX.append(tempX)
    
    for m in range(24):
        tempY.append(test_set[i+m][10])
    test_dataY.append(tempY)
    #print(i)
    i+=24
    
trainX = np.array(train_dataX)
trainY = np.array(train_dataY)
testX = np.array(test_dataX)
testY = np.array(test_dataY)


print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)



X = tf.placeholder(tf.float32, [None, x_dim])
Y = tf.placeholder(tf.float32, [None, 24])


#relu
W1 = tf.get_variable("W1", shape=[x_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([h_dim]), name='bias1')
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
layer1 = tf.nn.dropout(layer1, keep_prob=drop_out)

W2 = tf.get_variable("W2", shape=[h_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([h_dim]), name='bias2')
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob=drop_out)


W3 = tf.get_variable("W3", shape=[h_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([h_dim]), name='bias3')
layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)

W4 = tf.get_variable("W4", shape=[h_dim, h_dim], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([h_dim]), name='bias4')
layer4 = tf.nn.relu(tf.matmul(layer3, W4) + b4)

'''
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
hypothesis = tf.sigmoid(tf.matmul(layer4, W9) + b9)


# cost/loss function
loss = tf.reduce_mean(tf.square(hypothesis - Y))

train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# MAPE
targets = tf.placeholder(tf.float32, [None, 24])
predictions = tf.placeholder(tf.float32, [None, 24])
mape = tf.reduce_mean( tf.abs((targets - predictions)/targets) ) * 100


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i, step_loss))

    # test set Test step
    test_predict = sess.run(hypothesis, feed_dict={X: testX})
    mape_val = sess.run(mape, feed_dict={
                    targets: testY, predictions: test_predict})
    print("test set MAPE: {}".format(mape_val))
    
    # traing set Test step
    test_predict2 = sess.run(hypothesis, feed_dict={X: trainX})
    mape_val2 = sess.run(mape, feed_dict={
                    targets: trainY, predictions: test_predict2})
    print("train set MAPE: {}".format(mape_val2))



# Plot predictions
    plt.figure()
    plt.plot(trainY[0], label='original')
    plt.plot(test_predict2[0], label='prediction')
    plt.xlabel("Time Period")
    plt.ylabel("Electric Demand")
    plt.legend()
    plt.show()    

    
#%%
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
seq_length = 24
hseq_length = seq_length
data_dim = 10
hidden_dim = 10
output_dim = 24
learning_rate = 0.01
iterations = 10000
layer_num = 2
dropout = 1.0


#Open train_data, test_data
train_set = np.loadtxt('C:\exercise_data\\요일별\\Monday_hours_train.csv', delimiter=',', usecols=range(11))
test_set = np.loadtxt('C:\exercise_data\\요일별\\Monday_hours_test.csv', delimiter=',', usecols=range(11))


# build datasets
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    
    num = len(time_series)
    i=0
    while i <= num-hseq_length-24:
        _x = time_series[i:i+hseq_length, 1:11]
        _y = []
        for j in range(24):
            _y.append(time_series[i+hseq_length+j][-1])
        
        print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)
        
        i+=24
        
    return np.array(dataX), np.array(dataY)


trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length)
'''
print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)
print(len(trainX))
print(len(trainY))
print(len(testX))
print(len(testY))
'''


# input place holders
X = tf.placeholder(tf.float32, [None, hseq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 24])

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


cell = tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=dropout)
multi_cells = tf.contrib.rnn.MultiRNNCell([cell for _ in range(layer_num)], state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)

#print(type(outputs))
#print(outputs.shape)


Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 24])
predictions = tf.placeholder(tf.float32, [None, 24])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
# MAPE
mape = tf.reduce_mean( tf.abs((targets - predictions)/targets) ) * 100


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
    mape_val = sess.run(mape, feed_dict={
                    targets: testY, predictions: test_predict})
    print("test set MAPE: {}".format(mape_val))
    
    # train set Test step
    test_predict2 = sess.run(Y_pred, feed_dict={X: trainX})
    mape_val2 = sess.run(mape, feed_dict={
                    targets: trainY, predictions: test_predict2})
    print("train set MAPE: {}".format(mape_val2))

    # Plot predictions
    plt.figure()
    plt.plot(testY[0], label='original')
    plt.plot(test_predict[0], label='prediction')
    plt.xlabel("Time Period")
    plt.ylabel("Electric Demand")
    plt.legend()
    plt.show()


    
    
    
    
    
#%%
   
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()
tf.set_random_seed(777)  # for reproducibility



learning_rate = 0.001
past_load_num = 5*24
x_dim = 5+5*24+past_load_num
h_dim = 250
iterations = 1250
drop_out = 1.0


train_set = np.loadtxt('C:\exercise_data\\original_hours_month_1212_train_.csv', delimiter=',', usecols=range(12))
test_set = np.loadtxt('C:\exercise_data\\original_hours_month_1212_test_.csv', delimiter=',', usecols=range(12))
train_dataX = []
train_dataY = []
test_dataX = []
test_dataY = []


i = past_load_num
num = len(train_set)
while i <= num-24:
    for t in range(24):
        if train_set[i+t][11]==0:
            i+=24
            continue
    
    tempX = []
    tempY = []
    for n in range(24):
        for j in range(11):
            if j==0 :
                continue
            elif n==0 and j<=5:
                tempX.append(train_set[i+n][j])
            elif j>=6:
                tempX.append(train_set[i-24+n][j])
       
    
    for k in range(past_load_num):
        tempX.append(train_set[i-past_load_num+k][11])
    
    train_dataX.append(tempX)
    
    for m in range(24):
        tempY.append(train_set[i+m][11])
    train_dataY.append(tempY)
    #print(i)
    i+=24

i = past_load_num
num = len(test_set)
while i <= num-24:
    for t in range(24):
        if test_set[i+t][11]==0:
            i+=24
            continue
    
    tempX = []
    tempY = []
    for n in range(24):
        for j in range(11):
            if j==0:
                continue
            elif n==0 and j<=5:
                tempX.append(test_set[i+n][j])
            elif j>=6:
                tempX.append(test_set[i-24+n][j])
       
    
    for k in range(past_load_num):
        tempX.append(test_set[i-past_load_num+k][11])
    
    test_dataX.append(tempX)
    
    for m in range(24):
        tempY.append(test_set[i+m][11])
    test_dataY.append(tempY)
    #print(i)
    i+=24
    
trainX = np.array(train_dataX)
trainY = np.array(train_dataY)
testX = np.array(test_dataX)
testY = np.array(test_dataY)


print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)



X = tf.placeholder(tf.float32, [None, x_dim])
Y = tf.placeholder(tf.float32, [None, 24])


#relu
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
mape = tf.reduce_mean( tf.abs((targets - predictions)/targets) ) * 100


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
#plt.plot(iteration, label='iterations')
plt.plot(test_set_mape, label='test set')
plt.plot(train_set_mape, label='train set')
plt.xlabel("iterations")
plt.ylabel("MAPE(%)")
plt.legend()
plt.show()        
plt.clf()

    
    
    
#%%
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
seq_length = 5
hseq_length = 24*seq_length
data_dim = 11
hidden_dim = 15
output_dim = 24
learning_rate = 0.01
iterations = 2500
layer_num = 2
dropout = 1.0


#Open train_data, test_data
train_set = np.loadtxt('C:\exercise_data\\original_hours_month_91011_train_.csv', delimiter=',', usecols=range(12))
test_set = np.loadtxt('C:\exercise_data\\original_hours_month_91011_test_.csv', delimiter=',', usecols=range(12))


# build datasets
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    
    num = len(time_series)
    i=0
    while i <= num-hseq_length-24:
        for t in range(24):
            if time_series[i+hseq_length+t][11]==0:
                i+=24
                continue
            
        _x = time_series[i:i+hseq_length, 1:12]
        _y = []
        for j in range(24):
            _y.append(time_series[i+hseq_length+j][-1])
        
        print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)
        
        i+=24
        
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
X = tf.placeholder(tf.float32, [None, hseq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 24])

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


#cell = tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=dropout)
#multi_cells = tf.contrib.rnn.MultiRNNCell([cell for _ in range(layer_num)], state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(lstm_cell(), X, dtype=tf.float32)

#print(type(outputs))
#print(outputs.shape)


Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 24])
predictions = tf.placeholder(tf.float32, [None, 24])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
# MAPE
mape = tf.reduce_mean( tf.abs((targets - predictions)/targets) ) * 100


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
#plt.plot(iteration, label='iterations')
plt.plot(test_set_mape, label='test set')
plt.plot(train_set_mape, label='train set')
plt.xlabel("iterations")
plt.ylabel("MAPE(%)")
plt.legend()
plt.show() 
plt.clf()

# Plot predictions
plt.figure()
plt.plot(testY[0], label='original')
plt.plot(test_pred[0], label='prediction')
plt.xlabel("Time Period")
plt.ylabel("Electric Demand")
plt.legend()
plt.show()  
plt.clf()
'''
# Plot predictions
plt.figure()
plt.plot(testY[99], label='original')
plt.plot(test_pred[99], label='prediction')
plt.xlabel("Time Period")
plt.ylabel("Electric Demand")
plt.legend()
plt.show()  
plt.clf()
  
# Plot predictions
plt.figure()
plt.plot(testY[199], label='original')
plt.plot(test_pred[199], label='prediction')
plt.xlabel("Time Period")
plt.ylabel("Electric Demand")
plt.legend()
plt.show()  
plt.clf()  
'''
#%%
import math
pi = 3.14159265359
for i in range(12):
    month = (0.25*math.sin((2*pi/12)*(i+1))+0.25) + (0.25*math.cos((2*pi/12)*(i+1)) + 0.25)
    print(month)

#%%
import tensorflow as tf
import numpy as np
import matplotlib
import os
from sklearn.decomposition import PCA


tf.reset_default_graph()
tf.set_random_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

import matplotlib.pyplot as plt



# train Parameters
n_comp = 1
seq_length = 7
data_dim = n_comp+1
hidden_dim = 5
output_dim = 1
learning_rate = 0.005
iterations = 20000
layer_num = 2
dropout = 1.0


#Open train_data, test_data
train_set = np.loadtxt('C:\exercise_data\\train_value_min_max_scalar4_.csv', delimiter=',', usecols=range(10))
test_set = np.loadtxt('C:\exercise_data\\test_value_min_max_value4_.csv', delimiter=',', usecols=range(10))


# build datasets
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i + seq_length, :]
        _y = time_series[i + seq_length, [-1]]  # Next close price
        
        #print(_y.shape)
        if(_y!=0) :
            #pca = PCA(n_components=n_comp)
            #pca.fit(_x)
            #pca_x = pca.transform(_x)
            #pca_x_load = np.hstack([pca_x, time_series[i:i + seq_length, [-1]]])
            
            #print(pca_x_load, "->", _y)
            #dataX.append(pca_x_load)
            dataX.append(_x)
            dataY.append(_y)
    return np.array(dataX), np.array(dataY)


trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length)

print(train_set.shape)
print(test_set.shape)
print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)
print(len(trainX))
print(len(trainY))
print(len(testX))
print(len(testY))    
    

print(trainX[0].shape)
pca1 = PCA(n_components=7)
pca1.fit(trainX[0])
pca1_trainX = pca1.transform(trainX[0])
print(pca1_trainX.shape)
print(len(pca1_trainX))

print(pca1_trainX)
print(pca1.explained_variance_ratio_*100)
per_var1 = np.round(pca1.explained_variance_ratio_*100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var1)+1)]

plt.bar(x=range(1, len(per_var1)+1), height=per_var1, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.xlabel('Scree Plot')
plt.show()



'''
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


#cell = tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=dropout)
#multi_cells = tf.contrib.rnn.MultiRNNCell([cell for _ in range(layer_num)], state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(lstm_cell(), X, dtype=tf.float32)

#print(type(outputs))
#print(outputs.shape)
#print(outputs[:, -1])

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
mape = tf.reduce_mean( tf.abs((targets - predictions)/targets) ) * 100


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
#plt.plot(iteration, label='iterations')
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
plt.plot(test_pred, label='prediction')
plt.xlabel("Time Period")
plt.ylabel("Maximum Electric Power")
plt.legend()
plt.show()
'''

    
#%%
""" Auto Encoder Example.
Build a 2 layers auto-encoder with TensorFlow to compress images to a
lower latent space and then reconstruct them.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.01
num_steps = 30000
batch_size = 256

display_step = 1000
examples_to_show = 10

# Network Parameters
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features (the latent dim)
num_input = 784 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

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
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={X: batch_x})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()




#%%
import tensorflow as tf
import numpy as np

tf.reset_default_graph()
tf.set_random_seed(777)  # for reproducibility

import matplotlib.pyplot as plt

learning_rate = 0.001
past_load_num = 14
x_dim = 8+past_load_num
h_dim = 25
iterations = 10000
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
    if train_set[i][9]==0:
        i+=1
        continue
    
    tempX = []
    tempY = []
    for j in range(9):
        if j==0:
            continue
        elif j<4:
            tempX.append(train_set[i][j])
        else:
            tempX.append(train_set[i-1][j])
    
    for k in range(past_load_num):
        tempX.append(train_set[i-past_load_num+k][9])
    
    train_dataX.append(tempX)
    tempY.append(train_set[i][9])
    train_dataY.append(tempY)
    print(i)
    i+=1

i = past_load_num
num = len(test_set)
while i < num:
    if test_set[i][9]==0:
        i+=1
        continue
    
    tempX = []
    tempY = []
    for j in range(9):
        if j==0:
            continue
        elif j<4:
            tempX.append(test_set[i][j])
        else:
            tempX.append(test_set[i-1][j])
    
    for k in range(past_load_num):
        tempX.append(test_set[i-past_load_num+k][9])
    
    test_dataX.append(tempX)
    tempY.append(test_set[i][9])
    test_dataY.append(tempY)
    print(i)
    i+=1
    
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
num_hidden_1 = 11 # 1st layer num features
num_hidden_2 = 5 # 2nd layer num features (the latent dim)
num_input = x_dim 




X = tf.placeholder(tf.float32, [None, x_dim])
X2 = tf.placeholder(tf.float32, [None, x_dim])
Y = tf.placeholder(tf.float32, [None, 1])

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




#relu
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



# cost/loss function
loss = tf.reduce_mean(tf.square(hypothesis - Y))

train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# MAPE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
mape = tf.reduce_mean( tf.abs((targets - predictions)/targets) ) * 100


test_set_mape = []
train_set_mape = []
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1, AE_num_steps+1):
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
#plt.plot(iteration, label='iterations')
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
 
    
    
    
    
#%%


   
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()
tf.set_random_seed(777)  # for reproducibility



learning_rate = 0.001
past_load_num = 5*24
x_dim = 5+5*24+past_load_num
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
while i <= num-24:
    for t in range(24):
        if train_set[i+t][11]==0:
            i+=24
            continue
    
    tempX = []
    tempY = []
    for n in range(24):
        for j in range(11):
            if j==0 :
                continue
            elif n==0 and j<=5:
                tempX.append(train_set[i+n][j])
            elif j>=6:
                tempX.append(train_set[i-24+n][j])
       
    
    for k in range(past_load_num):
        tempX.append(train_set[i-past_load_num+k][11])
    
    train_dataX.append(tempX)
    
    for m in range(24):
        tempY.append(train_set[i+m][11])
    train_dataY.append(tempY)
    #print(i)
    i+=24

i = past_load_num
num = len(test_set)
while i <= num-24:
    for t in range(24):
        if test_set[i+t][11]==0:
            i+=24
            continue
    
    tempX = []
    tempY = []
    for n in range(24):
        for j in range(11):
            if j==0:
                continue
            elif n==0 and j<=5:
                tempX.append(test_set[i+n][j])
            elif j>=6:
                tempX.append(test_set[i-24+n][j])
       
    
    for k in range(past_load_num):
        tempX.append(test_set[i-past_load_num+k][11])
    
    test_dataX.append(tempX)
    
    for m in range(24):
        tempY.append(test_set[i+m][11])
    test_dataY.append(tempY)
    #print(i)
    i+=24
    
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
num_hidden_1 = 120 # 1st layer num features
num_hidden_2 = 60 # 2nd layer num features (the latent dim)
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





#relu
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
mape = tf.reduce_mean( tf.abs((targets - predictions)/targets) ) * 100


test_set_mape = []
train_set_mape = []
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1, AE_num_steps+1):
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
#plt.plot(iteration, label='iterations')
plt.plot(test_set_mape, label='test set')
plt.plot(train_set_mape, label='train set')
plt.xlabel("iterations")
plt.ylabel("MAPE(%)")
plt.legend()
plt.show()        
plt.clf()



#%%