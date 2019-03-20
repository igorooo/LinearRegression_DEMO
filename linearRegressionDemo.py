import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(suppress=True)

# THETA.T * X  (result M x 1 vector of hypothesis(values) for every m)


def hypothesis(THETA, X):
    return X@THETA


def J(THETA, X, y):
    m, n = X.shape
    sum = 0
    h = hypothesis(THETA, X)
    return (np.sum(np.square(h-y))) / 2*m


def gradientStep(THETA, X, y, alfa):
    TEMP = hypothesis(THETA, X) - y
    m, n = X.shape
    nTHETA = THETA[:]
    nTHETA = np.array(nTHETA).reshape(n, 1)

    for i in range(0, n):
        x = X[:, i]
        x = np.array(x)
        x = x.reshape((1, m))
        nTHETA[i] = THETA[i] - (alfa * (x.dot(TEMP))) / m
    return nTHETA


def normalEquationSolve(X, y):
    return ((np.linalg.inv((X.T)@X))@(X.T)@y)


def gradientDescentSolve(X, y, iterations=1000, alfa=0.000000001, epsilon=0.2):
    m, n = X.shape
    THETA = -10*np.ones((n, 1))
    for iter in range(0, iterations):
        THETA = gradientStep(THETA, X, y, alfa)
        if iter % 10 == 0:
            plt.plot(X[:, 1], hypothesis(THETA, X))
    # plt.show()
    return THETA


features_x = pd.read_csv(
    'https://raw.githubusercontent.com/tuanavu/coursera-stanford/master/machine_learning/lecture/week_2/v_octave_tutorial_week_2/featuresX.dat', header=None)

prices_y = pd.read_csv(
    'https://raw.githubusercontent.com/tuanavu/coursera-stanford/master/machine_learning/lecture/week_2/v_octave_tutorial_week_2/priceY.dat', header=None)

# 40 - 47 test set

XX = features_x.iloc[:, :].values
yy = prices_y.iloc[:, :].values

(L, C) = XX.shape

x0 = np.ones((L, 1))
XX = np.hstack((x0, XX))
X = XX[:40, :]
y = yy[:40, :]
X_test = XX[40:47, :]
y_test = yy[40:47, :]
X = np.array(X)
y = np.array(y)


Q1 = gradientDescentSolve(X, y)
print(Q1)
plt.plot(X[:, 1], hypothesis(Q1, X), color='red')

# Q = THETA

# normal equation THETA
Q = normalEquationSolve(X, y)
print("@@@")
print(Q)
"""
#COST FUNCT TEST
print(Q.shape)
nQ = np.ones((3, 1))
print(J(nQ, X, y))
print("##")
print(J(Q, X, y))
"""


y_res = hypothesis(THETA=Q, X=X_test)
yy_res = hypothesis(Q, X)


plt.xlabel('size m^2')
plt.ylabel('price $')

plt.plot(X[:, 1], y, 'ro', label="training set", color='blue')
#plt.plot(X_test[:, 1], y_test, 'ro', label="test set", color='red')
#plt.plot(X_test[:, 1], y_res, 'ro', label="res set", color='yellow')
#plt.plot(X[:, 1], yy_res, color='green')
plt.show()
