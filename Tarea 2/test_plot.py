import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

xs = [0]
ys = [0]
def animate(i):
    x = xs[-1] + 1
    y = x
    xs.append(float(x))
    ys.append(float(y))
    ax1.clear()
    ax1.plot(xs, ys)


ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()