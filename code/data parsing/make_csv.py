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