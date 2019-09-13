import math

pi = 3.14159265359
for i in range(12):
    month = (0.25 * math.sin((2 * pi / 12) * (i + 1)) + 0.25) + (0.25 * math.cos((2 * pi / 12) * (i + 1)) + 0.25)
    print(month)

# %%
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
data_dim = n_comp + 1
hidden_dim = 5
output_dim = 1
learning_rate = 0.005
iterations = 20000
layer_num = 2
dropout = 1.0

# Open train_data, test_data
train_set = np.loadtxt('C:\exercise_data\\train_value_min_max_scalar4_.csv', delimiter=',', usecols=range(10))
test_set = np.loadtxt('C:\exercise_data\\test_value_min_max_value4_.csv', delimiter=',', usecols=range(10))


# build datasets
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i + seq_length, :]
        _y = time_series[i + seq_length, [-1]]  # Next close price

        # print(_y.shape)
        if (_y != 0):
            # pca = PCA(n_components=n_comp)
            # pca.fit(_x)
            # pca_x = pca.transform(_x)
            # pca_x_load = np.hstack([pca_x, time_series[i:i + seq_length, [-1]]])

            # print(pca_x_load, "->", _y)
            # dataX.append(pca_x_load)
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
print(pca1.explained_variance_ratio_ * 100)
per_var1 = np.round(pca1.explained_variance_ratio_ * 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var1) + 1)]

plt.bar(x=range(1, len(per_var1) + 1), height=per_var1, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.xlabel('Scree Plot')
plt.show()
