import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.io as sio

data = loadmat('ex3data1.mat')
#print(data)

print(data['X'].shape, data['y'].shape)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y, learningRate):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg


def gradient_with_loop(theta, X, y, learningRate):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])

        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:, i])

    return grad


def gradient(theta, X, y, learningRate):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)

    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y

    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)

    # intercept gradient is not regularized
    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)

    return np.array(grad).ravel()



from scipy.optimize import minimize

def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]

    # k X (n + 1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))

    # insert a column of ones at the beginning for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])     #分别是训练1-10的theta（一次循环是一个数字）
        y_i = np.reshape(y_i, (rows, 1))
        #print("---",y_i.shape)              #共10个（5000，1）的数据
        # minimize the objective function
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradient)
        all_theta[i - 1, :] = fmin.x

    return all_theta


print(np.unique(data['y']))#看下有几类标签

all_theta = one_vs_all(data['X'], data['y'], 10, 1)
#print(all_theta)


def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # convert to matrices
    X = np.mat(X)
    all_theta = np.mat(all_theta)

    # compute the class probability for each class on each training instance
    h = sigmoid(X * all_theta.T)

    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)

    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1
    #print(h_argmax.shape)
    return h_argmax



y_pred = predict_all(data['X'], all_theta)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))




#################################################神经网络模型图示################################################


def load_weight(path):
    data = sio.loadmat(path)
    return data['Theta1'], data['Theta2']


theta1, theta2 = load_weight('ex3weights.mat')
#theta1.shape, theta2.shape
#((25, 401), (10, 26))      即第一层401个参数。第一隐藏层25个参数。第二隐藏层10个参数。

data = loadmat('ex3data1.mat')
#print(data)

X, y = data['X'],data['y']

X = np.insert(X, 0, values=np.ones(X.shape[0]), axis=1)  # intercept

print(X.shape, y.shape)


a1 = X

z2 = a1 @ theta1.T # (5000, 401) @ (25,401).T = (5000, 25)
print(z2.shape)#(5000, 25)

z2 = np.insert(z2, 0, values=np.ones(z2.shape[0]), axis=1)

a2 = sigmoid(z2)
print(a2.shape)#(5000, 26)


z3 = a2 @ theta2.T
print(z3.shape)#(5000, 10)

a3 = sigmoid(z3)
print(a3)

y_pred = np.argmax(a3, axis=1) + 1  # numpy is 0 base index, +1 for matlab convention，返回沿轴axis最大值的索引，axis=1代表行
print(y_pred.shape)#(5000, )


from sklearn.metrics import classification_report#这个包是评价报告

print(classification_report(y, y_pred))
#其中列表左边的一列为分类的标签名，右边support列为每个标签的出现次数．avg / total行为各列的均值（support列为总和）．
#precision recall f1-score三列分别为各个类别的精确度/召回率及 F1值．








