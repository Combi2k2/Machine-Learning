import numpy as np
import matplotlib.pyplot as plt
import math

from matplotlib.animation import FuncAnimation

X = np.random.rand(20,2)
one  = np.ones((X.shape[0],1))
Xbar = np.concatenate((one,X),axis = 1)

x0, y0 = (X.T)[0][:10], (X.T)[1][:10]   #belong to red class
x1, y1 = (X.T)[0][10:], (X.T)[1][10:]   #belong to blue class

for i in range(10):
    if (y0[i] < x0[i]): x0[i], y0[i] = y0[i], x0[i]
    if (y1[i] > x1[i]): x1[i], y1[i] = y1[i], x1[i]

w = [1,1,-1]

def animate(i):
    global w

    misclassified = []

    for i in range(10):
        v = x0[i] * w[0] + y0[i] * w[1] + w[2]
        if (v < 0):
            misclassified.append([ x0[i],  y0[i],  1])
    
    for i in range(10):
        v = x1[i] * w[0] + y1[i] * w[1] + w[2]
        if (v >= 0):
            misclassified.append([-x1[i], -y1[i], -1])

    if (len(misclassified) > 0):
        index = np.random.randint(0,len(misclassified))

        w[0] = w[0] + misclassified[index][0]
        w[1] = w[1] + misclassified[index][1]
        w[2] = w[2] + misclassified[index][2]
    
    wx = np.linspace(-5, 5, 100)
    wy = -(w[0] * wx + w[2]) / w[1]

    plt.cla()
    plt.plot(wx, wy, '-r', label = '{}X + {}Y + {} = 0'.format(w[0], w[1], w[2]))
    #plt.xlabel('x1')
    #plt.ylabel('x2')
    plt.legend(loc = 'upper left')

    plt.plot(x0, y0, 'ro')
    plt.plot(x1, y1, 'bs')

    plt.axis([0, 1, 0 ,1])

ani = FuncAnimation(plt.gcf(),animate, interval = 500)

plt.tight_layout()
plt.show()