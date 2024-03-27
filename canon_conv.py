import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from tqdm import tqdm


def lrelu(x):
    return x * (x > 0).astype("float32") + 0.2 * x * (x <= 0).astype("float32")


def DN(x, W1, B1, W2, B2):
    h1 = np.dot(x, W1) + B1
    u1 = lrelu(h1)
    h2 = np.dot(u1, W2) + B2
    return h1, h2


def DN_conv(X, W1, B1, W2, B2):
    """
    args:

    x : 2d image

    W1 : 3d tensor (n filters, height, width)

    B1 : vector (n filters)

    W2 : 4d tensor (n filters 2, n filters, height, width)

    B2 : vector (n filters 2)

    """

    H1 = []
    H2 = []

    for n in tqdm(range(len(X))):
        x = X[n]
        h1 = np.stack([convolve2d(x, W1[k]) for k in range(W1.shape[0])]) + B1
        u1 = lrelu(h1)
        h2 = (
            np.stack(
                [
                    np.stack(
                        [
                            convolve2d(u1[c], W2[k, c])
                            for c in range(W2.shape[1])
                        ]
                    ).sum(0)
                    for k in range(W2.shape[0])
                ]
            )
            + B2
        )
        H1.append(h1)
        H2.append(h2)
    return np.stack(H1), np.stack(H2)


L = 2
N = 200
grid = np.meshgrid(np.linspace(-L, L, N), np.linspace(-L, L, N))
x = np.stack([grid[0].reshape(-1), grid[1].reshape(-1)], 1)
h = DN(
    x,
    np.random.randn(2, 6),
    np.random.randn(6),
    np.random.randn(6, 3),
    np.random.randn(3),
)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
for k in range(6):
    plt.contour(
        grid[0], grid[1], h[0][:, k].reshape((N, N)), levels=[0], colors="b"
    )
plt.xticks([])
plt.yticks([])
plt.title("layer1")

plt.subplot(1, 3, 2)
for k in range(3):
    plt.contour(
        grid[0], grid[1], h[1][:, k].reshape((N, N)), levels=[0], colors="r"
    )
plt.xticks([])
plt.yticks([])
plt.title("layer2")

plt.subplot(1, 3, 3)
for k in range(6):
    plt.contour(
        grid[0], grid[1], h[0][:, k].reshape((N, N)), levels=[0], colors="b"
    )
for k in range(3):
    plt.contour(
        grid[0], grid[1], h[1][:, k].reshape((N, N)), levels=[0], colors="r"
    )
plt.xticks([])
plt.yticks([])
plt.title("layer1+2")

plt.savefig("mini_test.png")


N = 200
T = 8
image_to_study = np.random.randn(T, T)
direction = np.random.randn(T, T)

alpha, beta = np.meshgrid(np.linspace(0, 2, N), np.linspace(-1, 1, N))
alpha = alpha.reshape(-1)[:, None, None]
beta = beta.reshape(-1)[:, None, None]

x = alpha * image_to_study + beta * direction

h = DN_conv(
    x,
    np.random.randn(4, 3, 3),
    np.random.randn(4, 1, 1) / 10,
    np.random.randn(3, 4, 3, 3),
    np.random.randn(3, 1, 1) / 10,
)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
for k in range(4):
    for i in range(T - 2):
        for j in range(T - 2):
            plt.contour(
                alpha.reshape((N, N)),
                beta.reshape((N, N)),
                h[0][:, k, i, j].reshape((N, N)),
                levels=[0],
                colors="b",
            )
plt.xticks([])
plt.yticks([])
plt.title("layer1")

plt.subplot(1, 2, 2)
for k in range(3):
    for i in range(T - 4):
        for j in range(T - 4):
            plt.contour(
                alpha.reshape((N, N)),
                beta.reshape((N, N)),
                h[1][:, k, i, j].reshape((N, N)),
                levels=[0],
                colors="b",
            )
plt.xticks([])
plt.yticks([])
plt.title("layer2")

plt.savefig("mini_conv.png")
