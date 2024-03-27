import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
from sklearn.datasets import make_blobs
import torch.nn.utils.prune as prune
import json

torch.manual_seed(42)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(2, 6)
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x):
        l1 = self.fc1(x)
        l2 = self.fc2(F.leaky_relu(l1, negative_slope=0.2))
        return l1, l2


def prune_model(model, pruning_strategy, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            pruning_method = getattr(prune, pruning_strategy)
            pruning_method(module, name="weight", amount=amount)
    return model


X_train, y_train = make_blobs(n_samples=500, n_features=2, centers=3)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.int64)

with open("config.json") as f:
    model_specs = json.load(f)["models"]

models = []
model_names = []
for model in model_specs:
    model_names.append(model["name"])
    models.append(prune_model(MLP(), model["pruning_strategy"], model["pruning_level"]))

num_models = len(models)
optimizers = [optim.SGD(model.parameters(), lr=0.1) for model in models]
criterion = nn.CrossEntropyLoss()

L = 10
N = 1000


grid = np.meshgrid(np.linspace(-L, L, N), np.linspace(-L, L, N))
x_grid = np.stack([grid[0].reshape(-1), grid[1].reshape(-1)], axis=1)
x_grid_torch = torch.tensor(x_grid, dtype=torch.float32)

epochs = 100
for epoch in range(epochs):
    for model, optimizer in zip(models, optimizers):
        _, y_hat = model(X_train)
        loss = criterion(y_hat, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0 or epoch == epochs - 1:
        fig, axs = plt.subplots(num_models + 1, 4, figsize=(20, 5 * (num_models + 1)))
        for i, (model, model_name) in enumerate(zip(models, model_names)):

            h1, h2 = model(x_grid_torch)
            h1, h2 = h1.detach().numpy(), h2.detach().numpy()

            for k in range(6):
                axs[i, 0].contour(
                    grid[0], grid[1], h1[:, k].reshape((N, N)), levels=[0], colors="b"
                )
            axs[i, 0].set_title("layer1: " + model_name)

            for k in range(3):
                axs[i, 1].contour(
                    grid[0], grid[1], h2[:, k].reshape((N, N)), levels=[0], colors="r"
                )
            axs[i, 1].set_title("layer2: " + model_name)

            for k in range(6):
                axs[i, 2].contour(
                    grid[0], grid[1], h1[:, k].reshape((N, N)), levels=[0], colors="b"
                )
            for k in range(3):
                axs[i, 2].contour(
                    grid[0], grid[1], h2[:, k].reshape((N, N)), levels=[0], colors="r"
                )
            axs[i, 2].set_title("layer1+2: " + model_name)

            predictions = torch.argmax(model(X_train)[1], dim=1).detach().numpy()
            axs[i, 3].scatter(X_train[:, 0], X_train[:, 1], c=predictions, s=30)
            axs[i, 3].set_title("Predicted classes: " + model_name)

        axs[num_models, 3].scatter(
            X_train[:, 0], X_train[:, 1], c=y_train.numpy(), s=30
        )
        axs[num_models, 3].set_title("Actual classes")
        for ax in axs[num_models, :-1].flatten():
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"plt/mlp/synthetic/l1_{epoch}.png")
        plt.close(fig)
