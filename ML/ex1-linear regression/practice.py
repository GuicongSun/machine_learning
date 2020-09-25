import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import mat
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])

print(data.describe())

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
plt.show()



data.insert(0, 'Ones', 1)
#print(data.head())

cols = data.shape[1]
X = data.iloc[:,0:cols-1]#X是所有行，去掉最后一列
y = data.iloc[:,cols-1:cols]#X是所有行，最后一列

#print(X.head())#head()是观察前5行
#print(y.head())

X = mat(X)
y = mat(y)
theta = mat(np.zeros([1,2]))
print(X.shape,y.shape,theta.shape)
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

print(computeCost(X, y, theta))

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.array(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])        #ravel扁平化函数   shape[1]代表扁平化后有几列就有几个theta（自变量）
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost

alpha = 0.01
iters = 1000

g, cost = gradientDescent(X, y, theta, alpha, iters)
print(g.ravel())
print(computeCost(X, y, g))

x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()



'''
#正规方程法
def normalEqn(X, y):
    theta = np.linalg.inv(X.T@X)@X.T@y#X.T@X等价于X.T.dot(X)
    return theta

final_theta2=normalEqn(X, y)#感觉和批量梯度下降的theta的值有点差距
print(final_theta2)
print(final_theta2.shape)
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = final_theta2[0, 0] + (final_theta2[1, 0] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()'''

'''
#通过sklearn函数库实现。
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X, y)

x = np.array(X[:, 1].A1)
f = model.predict(X).flatten()

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()'''

## 多变量线性回归

path = 'ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
data2.head()

data2 = (data2 - data2.mean()) / data2.std()
data2.head()

# add ones column
data2.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data2.shape[1]
X2 = data2.iloc[:,0:cols-1]
y2 = data2.iloc[:,cols-1:cols]

# convert to matrices and initialize theta
X2 = mat(X2.values)
y2 = mat(y2.values)
theta2 = mat(np.array([0,0,0]))

print(computeCost(X2, y2, theta2))

# perform linear regression on the data set
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)
print(g2.ravel())
# get the cost (error) of the model
print(computeCost(X2, y2, g2))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(data2.Size, data2.Bedrooms,data2.Price, label='Traning Data')
x = np.linspace(data2.Size.min(), data2.Size.max(), 100)
y = np.linspace(data2.Bedrooms.min(), data2.Bedrooms.max(), 100)
z = g2[0, 0] + (g2[0, 1] * x)+ (g2[0, 2] * y)
ax.plot(x, y,z, 'r', label='Prediction')
ax.set_xlabel('Size')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')
ax.set_title('lallala')
plt.show()

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()




