import numpy as np
import matplotlib.pyplot as plt
import math

N = 20

X = np.linspace(-3, 12, N) + np.random.randn(N)
y = np.array([X ** 3 - 5 * X ** 2 - 10 * X + 3 + np.random.randn(N) * 30])

plt.plot(X, y[0], 'ro')
plt.xlim([-4, 13])
plt.ylim([-200, 1000])
plt.show()

# Building Xbar

Xbar = np.ones((N, 4))

for i in range(N):
    for j in range(4):
        Xbar[i, j] = X[i] ** j

print(Xbar)

#y = Xbar * w -> Find argmin w

A = np.dot(Xbar.T, Xbar)   #   matrix multiplication
b = np.dot(Xbar.T, y.T)

#print('A = ',A)

w = np.dot(np.linalg.pinv(A),b) #   w is product of pseudo inverse of A and b

print('w = ',w)

X_predicted = np.linspace(-4, 13, 100)
Y_predicted = np.linspace( 0,  0, 100)

for i in range(4):
    Y_predicted += X_predicted ** i * w[i]

plt.plot(X_predicted, Y_predicted, 'g-')
plt.plot(X, y[0], 'ro')

plt.xlim([-4, 13])
plt.ylim([-200, 1000])
plt.show()