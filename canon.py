import numpy as np
import matplotlib.pyplot as plt


def lrelu(x):
    return x * (x > 0).astype('float32') + 0.2 * x * (x <= 0).astype('float32')

def DN(x, W1, B1, W2, B2):
    h1 = np.dot(x, W1)+B1
    u1 = lrelu(h1)
    h2 = np.dot(u1, W2) + B2
    return h1, h2


L = 2
N = 200
grid = np.meshgrid(np.linspace(-L, L, N),
                 np.linspace(- L, L, N))
x = np.stack([grid[0].reshape(-1), grid[1].reshape(-1)], 1)
h = DN(x, np.random.randn(2,6), np.random.randn(6),
        np.random.randn(6,3), np.random.randn(3))

plt.figure(figsize=(12,4))

plt.subplot(1, 3, 1)
for k in range(6):
    plt.contour(grid[0], grid[1], h[0][:, k].reshape((N, N)), levels=[0], colors='b')
plt.xticks([])
plt.yticks([])
plt.title('layer1')

plt.subplot(1, 3, 2)
for k in range(3):
    plt.contour(grid[0], grid[1], h[1][:, k].reshape((N, N)), levels=[0], colors='r')
plt.xticks([])
plt.yticks([])
plt.title('layer2')

plt.subplot(1, 3, 3)
for k in range(6):
    plt.contour(grid[0], grid[1], h[0][:, k].reshape((N, N)), levels=[0], colors='b')
for k in range(3):
    plt.contour(grid[0], grid[1], h[1][:, k].reshape((N, N)), levels=[0], colors='r')
plt.xticks([])
plt.yticks([])
plt.title('layer1+2')

plt.savefig('mini_test.png')
