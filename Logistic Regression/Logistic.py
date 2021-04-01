import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from matplotlib.animation import FuncAnimation

np.random.seed(2)

hour = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
Pass = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

hour = np.concatenate((np.ones((1, hour.shape[1])), hour), axis = 0)

def sigmoid(s):
    return  1 / (1 + np.exp(-s))

X0 = hour[1, np.where(Pass == 0)][0]
y0 = Pass[np.where(Pass == 0)]

X1 = hour[1, np.where(Pass == 1)][0]
y1 = Pass[np.where(Pass == 1)]

N = hour.shape[1]
d = hour.shape[0]

w = [np.random.randn(d, 1)]

def Logistic(i):
    mix_id = np.random.permutation(N)

    for i in mix_id:
        xi = hour[:, i].reshape(d, 1)
        yi = Pass[i]
        zi = sigmoid(np.dot(w[-1].T, xi))
        w_new = w[-1] + .05 * (yi - zi)*xi

        w.append(w_new)
    
    wx = np.linspace(0, 6, 1000)
    wy = sigmoid(w[-1][0] + wx * w[-1][1])

    plt.cla()
    plt.plot(X0, y0, 'ro')
    plt.plot(X1, y1, 'bs')

    plt.plot(wx, wy, 'g-', linewidth = 2)

    plt.xlabel('studying hours')
    plt.ylabel('predicted probability of pass')
    
    plt.axis([0, 6, -0.5, 5])

ani = FuncAnimation(plt.gcf(), Logistic, interval = 500)

plt.tight_layout()
plt.show()
