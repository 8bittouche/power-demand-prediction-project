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


f = open('C:\exercise_data\date_20060101_20180930___test.csv', 'r')
f2 = open('C:\exercise_data\date_20060101_20180930___test3.csv', 'w')
while True:
    # line = f.read().split(',')
    line = f.readline()
    if not line: break
    data = line.split(',')
    # print(line.shape)
    # print(lines)
    # data = line[:]
    # print(data[0])
    # print(data[1])
    # print(data[2])
    # print(data[3])
    # print(data[4])

    if data[3] == '토':
        print(data[3])
        data[4] = 1
    elif data[3] == '일':
        data[4] = 1
        print(data[3])

    newline = str(data[0]) + ',' + str(data[1]) + ',' + str(data[2]) + ',' + str(data[3]) + ',' + str(
        data[4]) + ',' + '\n'
    print(newline)
    f2.write(newline)

f.close()

f = open('C:\exercise_data\\data_real_last.csv', 'r')
f2 = open('C:\exercise_data\\data_real_last2.csv', 'w')
while True:
    line = f.readline()
    if not line: break
    # data = line.split(',')
    # print(data)

    if data[5] == '':
        print(data[0] + '년' + data[1] + '월' + data[2] + '일')
    if data[6] == '':
        print(data[0] + '년' + data[1] + '월' + data[2] + '일')
    if data[7] == '':
        print(data[0] + '년' + data[1] + '월' + data[2] + '일')
    if data[8] == '':
        print(data[0] + '년' + data[1] + '월' + data[2] + '일')
    if data[9] == '':
        print(data[0] + '년' + data[1] + '월' + data[2] + '일')
    if data[10] == '':
        print(data[0] + '년' + data[1] + '월' + data[2] + '일')
    if data[11] == '':
        print(data[0] + '년' + data[1] + '월' + data[2] + '일')
    if data[12] == '':
        print(data[0] + '년' + data[1] + '월' + data[2] + '일')

    data = line[:-1] + "," + "\n"
    f2.write(data)
f.close()


data = []
pi = 3.14159265359
xy = np.loadtxt('C:\exercise_data\\month_day_dayofweek3.csv', delimiter=',', usecols=range(3))
for i in range(len(xy)):
    temp = []
    # print(xy[i][1])
    if i < len(xy) - 30:
        if xy[i][1] == 1:
            if xy[i + 27][1] == 1:
                month_len = 28
            elif xy[i + 28][1] == 1:
                month_len = 29
            elif xy[i + 29][1] == 1:
                month_len = 30
            elif xy[i + 30][1] == 1:
                month_len = 31
    else:
        month_len = 30
    print(month_len)

    month = (0.25 * math.sin((2 * pi / 12) * xy[i][0]) + 0.25) + (0.25 * math.cos((2 * pi / 12) * xy[i][0]) + 0.25)
    day = (0.25 * math.sin((2 * pi / month_len) * xy[i][1]) + 0.25) + (
                0.25 * math.cos((2 * pi / month_len) * xy[i][1]) + 0.25)
    dayOfweek = (0.25 * math.sin((2 * pi / 7) * xy[i][2]) + 0.25) + (0.25 * math.cos((2 * pi / 7) * xy[i][2]) + 0.25)

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
    if s[10] == '0' and s[11] == '0' and s[12] == '0' and s[13] == '0':
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

    x1 = 0.08 * gwangju[i][7] + 0.12 * daegu[i][7] + 0.1 * daejeon[i][7] + 0.2 * busan[i][7] + 0.5 * seoul[i][7]
    x2 = 0.08 * gwangju[i][8] + 0.12 * daegu[i][8] + 0.1 * daejeon[i][8] + 0.2 * busan[i][8] + 0.5 * seoul[i][8]
    x3 = 0.08 * gwangju[i][9] + 0.12 * daegu[i][9] + 0.1 * daejeon[i][9] + 0.2 * busan[i][9] + 0.5 * seoul[i][9]
    x4 = 0.08 * gwangju[i][10] + 0.12 * daegu[i][10] + 0.1 * daejeon[i][10] + 0.2 * busan[i][10] + 0.5 * seoul[i][10]
    x5 = 0.08 * gwangju[i][11] + 0.12 * daegu[i][11] + 0.1 * daejeon[i][11] + 0.2 * busan[i][11] + 0.5 * seoul[i][11]

    if i < len(gwangju) - 30 * 24:
        if gwangju[i][3] == 1 and gwangju[i][4] == 0:
            if gwangju[i + 28 * 24][3] == 1:
                month_len = 28
            elif gwangju[i + 29 * 24][3] == 1:
                month_len = 29
            elif gwangju[i + 30 * 24][3] == 1:
                month_len = 30
            elif gwangju[i + 31 * 24][3] == 1:
                month_len = 31
    else:
        month_len = 30

    year = gwangju[i][1]
    month = (0.25 * math.sin((2 * pi / 12) * gwangju[i][2]) + 0.25) + (
                0.25 * math.cos((2 * pi / 12) * gwangju[i][2]) + 0.25)
    day = (0.25 * math.sin((2 * pi / month_len) * gwangju[i][3]) + 0.25) + (
                0.25 * math.cos((2 * pi / month_len) * gwangju[i][3]) + 0.25)
    hours = (0.25 * math.sin((2 * pi / 24) * gwangju[i][4]) + 0.25) + (
                0.25 * math.cos((2 * pi / 24) * gwangju[i][4]) + 0.25)
    dayOfweek = (0.25 * math.sin((2 * pi / 7) * gwangju[i][5]) + 0.25) + (
                0.25 * math.cos((2 * pi / 7) * gwangju[i][5]) + 0.25)
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