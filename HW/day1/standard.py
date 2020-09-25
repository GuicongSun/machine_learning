import pandas as pd
import numpy as np
import sys


#数据预处理
data = pd.read_csv('./train.csv', encoding = 'big5')
data = data. iloc[:,3: ]
data[data=='NR'] = 0

raw_data = np.array(data) # DataFrame转换成numpy数组
print(raw_data.shape)
#print(raw_data)

month_data = {} # key: month value: data
for month in range(12):
    sample = np.empty([18, 480]) # 创建一个空的[18，480] 数组
    for day in range(20):
        sample[:,day*24:(day+1)*24]=raw_data[18*(20*month+day):18*(20*month+day+1),:]
    month_data[month] = sample

#以第一个月为例
print(month_data[0].shape)


x = np. empty([12 * 471, 18 * 9], dtype = float)
y = np. empty([12 * 471, 1],dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day==19 and hour>14:
                continue
            #reshape将矩阵重整为新的行列数，参数-1代表自动推断,这里去掉了18*9的二维属性，
            #转而以一维序列代替，一维序列的顺序本身可以隐含其时序信息
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] #value
print(x.shape)
print(y.shape)
#print(x)
#print(y)
x_vari=x[5300:,:]
y_vari=y[5300:,:]
x=x[:5300,:]
y=y[:5300,:]

mean_x = np.mean(x, axis = 0) #18 * 9  求均值，aix=0表示沿每列计算
std_x = np.std(x, axis = 0) #18 * 9  标准差
for i in range(len(x)): #12 * 471
    for j in range(len(x[0])): #18 * 9
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]  #所有属性归一化，避免使数据的某些特征形成主导作用


dim = x.shape[1] + 1
w = np.zeros(shape = (dim, 1 )) #empty创建的数组， 数组中的数取决于数组在内存中的位置处的值，为o纯属巧合?
x = np.concatenate( (np . ones((x.shape[0], 1 )), x) , axis = 1).astype(float)
#初始化学习率(163个参数，163个200 )和adagrad
learning_rate = np.array([[100]] * dim)
adagrad_sum = np.zeros(shape = (dim, 1 ))
#print(learning_rate.size)
#print(learning_rate.shape)
#没有隐藏层的网络
for T in range(10001):
    if(T%500==0):
        print("T=",T)
        print("Loss:" ,np. sum((x.dot(w) - y)**2)/ x.shape[0] /2) #最小二乘损失
        print((x.dot(w) - y)**2)
    gradient = 2 * np. transpose(x) . dot(x. dot(w)-y) #损失的导数x*(yh-h)
    adagrad_sum += gradient ** 2
   # print(learning_rate)
    w = w- learning_rate * gradient / (np. sqrt(adagrad_sum) + 0.0005 )
np.save('weight.npy' ,w)

# dim = 18 * 9 + 1
# w = np.zeros([dim, 1])
# x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(float)
# learning_rate = 100
# iter_time = 1000
# adagrad = np.zeros([dim, 1])
# eps = 0.0000000001
# for t in range(iter_time):
#     loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12)  # rmse
#     if (t % 100 == 0):
#         print(str(t) + ":" + str(loss))
#     gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)  # dim*1
#     adagrad += gradient ** 2
#     w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
# np.save('weight.npy', w)

#验证 verification

x_vari = np.concatenate((np.ones([x_vari.shape[0], 1]), x_vari), axis = 1).astype(float)
print("Loss:",np.sum((x_vari.dot(w) - y_vari)**2)/ x_vari.shape[0] /2)
for i in range(y_vari.shape[0]):
    print(x_vari.dot(w)-y_vari)

'''
#通过测试集测试
testdata = pd.read_csv('./test.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

W = np. load( ' weight.npy ' )
ans_y = np.dot(test_x,w)        #预测得到结果

import csv
with open( ' submit.csv', mode='w', newline='' ) as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print (header)
    csv_writer. writerow( header )
    for i in range(240):
        row = ['id. ' + str(i), ans_y[i][0]]
        csv_writer. writerow(row)
        print(row)
'''