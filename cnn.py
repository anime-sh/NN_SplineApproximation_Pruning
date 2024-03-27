import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.utils.prune as prune
import json
import copy
from pytorch_lightning import seed_everything

seed_everything(42)


class ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool2d(2),
        )
        self.layer3 = nn.Linear(3 * 5 * 5, 10)

    def forward(self, x):
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2.reshape(x.shape[0], -1))
        return l1, l2, l3


def prune_model(model, pruning_strategy, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            pruning_method = getattr(prune, pruning_strategy)
            pruning_method(module, name="weight", amount=amount)
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("config_cnn.json") as f:
    model_specs = json.load(f)["models"]

models = []
model_names = []
for model in model_specs:
    model_names.append(model["name"])
    models.append(ConvNet().to(device))

num_models = len(models)
optimizers = [optim.SGD(model.parameters(), lr=0.1) for model in models]
criterion = nn.CrossEntropyLoss()

transform = transforms.ToTensor()

train_data = datasets.MNIST(
    root="~/MNIST", train=True, transform=transform, download=True
)
val_data = datasets.MNIST(
    root="~/MNIST", train=False, transform=transform, download=True
)

train_loader = DataLoader(dataset=train_data, batch_size=100, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=100)


N = 200
T = 28
image_to_study = torch.randn(1, 1, T, T).to(device)
direction = torch.randn(1, 1, T, T).to(device)

alpha, beta = torch.meshgrid(
    torch.linspace(0, 2, N).to(device), torch.linspace(-1, 1, N).to(device)
)
alpha = alpha.reshape(-1, 1, 1, 1)
beta = beta.reshape(-1, 1, 1, 1)

grid = alpha * image_to_study + beta * direction

epochs = 25
for epoch in range(epochs):
    if epoch % 5 == 0 and epoch:
        for i, spec in enumerate(model_specs):
            if spec["pruning_strategy"] == "lth":
                # models[i] = prune_model_lth(
                #     models[i], model_init_states[i], spec["pruning_level"]
                # )
                pass
            else:
                models[i] = prune_model(
                    models[i], spec["pruning_strategy"], spec["pruning_level"]
                )

    for model, optimizer in zip(models, optimizers):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            _, _, y_hat = model(images)
            loss = criterion(y_hat, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if epoch % 5 == 0 or epoch == epochs - 1:

        for model, model_name in zip(models, model_names):
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    _, _, outputs = model(images)
                    pred = torch.max(outputs.data, 1).indices
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()
            print(f"Epoch {epoch}, {model_name=} Val acc = {correct/total}")

        fig, axs = plt.subplots(num_models + 1, 2, figsize=(20, 5 * (num_models)))
        for i, (model, model_name) in enumerate(zip(models, model_names)):

            with torch.no_grad():
                h1, h2, _ = model(grid)
                h1, h2 = h1.cpu().detach().numpy(), h2.cpu().detach().numpy()

            for k in range(4):
                for i1 in range(T // 2 - 1):
                    for j in range(T // 2 - 1):
                        if (k + i1 + j) % 10 == 0:
                            axs[i, 0].contour(
                                alpha.cpu().numpy().reshape((N, N)),
                                beta.cpu().numpy().reshape((N, N)),
                                h1[:, k, i1, j].reshape((N, N)),
                                levels=[0],
                                colors="b",
                            )

            axs[i, 0].set_title("layer1: " + model_name)

            for k in range(3):
                for i1 in range((T // 4) - 2):
                    for j in range((T // 4) - 2):
                        if (k + i1 + j) % 5 == 0:
                            axs[i, 1].contour(
                                alpha.cpu().numpy().reshape((N, N)),
                                beta.cpu().numpy().reshape((N, N)),
                                h2[:, k, i1, j].reshape((N, N)),
                                levels=[0],
                                colors="r",
                            )
            axs[i, 1].set_title("layer2: " + model_name)
        for ax in axs[num_models, :].flatten():
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(f"plt/conv/mnist/l1_every5th_{epoch}.png")
        plt.close(fig)
