import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

Data = pd.read_csv('bottle.csv')

# Saltinity
X = np.array(Data['Salnty'])

# Temperature (deg C)
y = np.array(Data['T_degC'])

#Filled in deficient position:

len = X.shape[0]

X_mean = np.mean([v for v in X if math.isnan(v) == False])
y_mean = np.mean([v for v in y if math.isnan(v) == False])

for i in range(len):
    if (math.isnan(X[i]) == True):  X[i] = X_mean
    if (math.isnan(y[i]) == True):  y[i] = y_mean

print(math.isnan(X_mean))
print(math.isnan(y_mean))

X = np.array([X]).T
y = np.array([y]).T

# Visualize data 

plt.plot(X, y, 'ro')
plt.axis([0, 40, 0 ,100])
plt.xlabel('Saltinity')
plt.ylabel('Temperature')
plt.show()

# Building Xbar

one  = np.ones((X.shape[0],1))
Xbar = np.concatenate((one,X),axis = 1)

#y = Xbar * w -> Find argmin w

A = np.dot(Xbar.T, Xbar)   #   matrix multiplication
b = np.dot(Xbar.T, y)

#print('A = ',A)

w = np.dot(np.linalg.pinv(A),b) #   w is product of pseudo inverse of A and b

print('w = ',w)

