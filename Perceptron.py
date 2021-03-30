import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from matplotlib.animation import FuncAnimation

np.random.seed(2)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis = 1)
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)
# Xbar 
X = np.concatenate((np.ones((1, 2*N)), X), axis = 0)

def h(w, x):    
    return np.sign(np.dot(w.T, x))

def has_converged(X, y, w):    
    return np.array_equal(h(w, X), y) 

N = X.shape[1]
d = X.shape[0]

w = [np.random.randn(d, 1)]
mispoint = []

def perceptron(i):
    mix_id = np.random.permutation(N)

    for i in range(N):
        xi = X[:, mix_id[i]].reshape(d, 1)
        yi = y[0, mix_id[i]]

        if (h(w[-1], xi)[0] != yi): #misclassified point
            mis_points.append(mix_id[i])
            w_new = w[-1] + yi * xi
            w.append(w_new)
    
    wx = np.linspace(-5, 5, 100)
    wy = -(w[-1][0] * wx + w[-1][2]) / w[-1][1]

    plt.cla()
    plt.plot(wx, wy, '-r', label = '{}X + {}Y + {} = 0'.format(w[-1][0], w[-1][1], w[-1][2]))
    #plt.xlabel('x1')
    #plt.ylabel('x2')
    plt.legend(loc = 'upper left')

    plt.plot((X0.T)[0], (X0.T)[1], 'ro')
    plt.plot((X1.T)[0], (X1.T)[1], 'bs')

    plt.axis([0, 1, 0 ,1])

ani = FuncAnimation(plt.gcf(),perceptron, interval = 500)

plt.tight_layout()
plt.show()
