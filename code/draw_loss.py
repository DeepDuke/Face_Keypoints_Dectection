import numpy as np
from matplotlib import pyplot as plt


train_losses = np.loadtxt('train_losses.csv', delimiter=',')
valid_losses = np.loadtxt('valid_losses.csv', delimiter=',')

fig = plt.figure('loss')
ax1 = fig.add_axes()

length = len(train_losses)
epochs = list(range(length))
plt.plot(epochs, train_losses, color='b', linestyle='-', lw='1', marker='*', markersize=3, label='train-loss')
plt.plot(epochs, valid_losses, color='g', linestyle='-', lw='1', marker='d', markersize=3, label='valid-loss')
plt.legend()
plt.show()