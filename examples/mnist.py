import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from edl_pytorch import Dirichlet, evidential_classification

MEAN, STD = 0.13, 0.31


# re-create mnist example from http://arxiv.org/abs/1806.01768
def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    train_ds = MNIST("./data", train=True, download=True, transform=transform)
    test_ds = MNIST("./data", train=False, download=True, transform=transform)

    lenet = nn.Sequential(
        nn.Conv2d(1, 6, 5, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(6, 16, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        Dirichlet(84, 10),
    )
    lenet.to(device)

    optimizer = torch.optim.Adam(lenet.parameters(), lr=1e-3)

    for epoch in range(10):
        lenet.train()
        for x, y in DataLoader(train_ds, batch_size=1000, shuffle=True):
            x = x.to(device)
            y = y.to(device)
            pred = lenet(x)
            loss = evidential_classification(pred, y, lamb=min(1, epoch / 10))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            lenet.eval()
            correct, total = 0, 0
            for x, y in DataLoader(test_ds, batch_size=1000):
                x = x.to(device)
                y = y.to(device)
                pred = lenet(x)
                correct += (pred.argmax(-1) == y).sum()
                total += y.shape[0]

            acc = (correct / total).item()
            print("epoch:", epoch, "acc:", acc)

    # show uncertainty with rotation with one of the "1" examples
    rotation_uncertainty(test_ds[2][0].to(device), lenet)


@torch.no_grad()
def rotation_uncertainty(img, model):
    m = 180
    n = m // 10 + 1
    degs = torch.linspace(0, m, n)
    imgs, probs, uncertainty = [], [], []
    for deg in degs:
        img_ = TF.rotate(img, float(deg), fill=-MEAN / STD)
        imgs.append(img_.squeeze().cpu())
        alpha = model(img_.unsqueeze(0)).squeeze().cpu()
        probs.append(alpha / alpha.sum())
        uncertainty.append(10 / alpha.sum())

    uncertainty = torch.stack(uncertainty)
    probs = torch.stack(probs)
    imgs = torch.cat(imgs, dim=1)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [8, 1]}, figsize=(5, 3), dpi=200
    )
    fig.subplots_adjust(hspace=0.5)
    ax1.plot(degs, uncertainty, label="Unc.", marker="s")
    ax1.plot(degs, probs[:, 1], label="1", color="k", marker="s")
    ax1.set_ylabel("Probability")
    ax1.set_xlabel("Rotation Degrees")
    ax1.legend()

    ax2.imshow(1 - (imgs * STD + MEAN), cmap="gray")
    ax2.axis(False)

    plt.savefig(f"examples/mnist.png")


if __name__ == "__main__":
    main()
