from pickletools import optimize
from cv2 import transform
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import sys
import torchvision.datasets as datasets
np.set_printoptions(threshold=sys.maxsize)
transform2 = torchvision.transforms.Compose([torchvision.transforms.ToTensor(
), torchvision.transforms.Normalize((0.5,), (0.5,)), ])
mnist_trainset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform2)
mnist_testset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform2)
train1 = DataLoader(mnist_trainset, shuffle=True)
test1 = DataLoader(mnist_testset, shuffle=True)



device = "cuda" if torch.cuda.is_available() else "cpu"
print("I am using {} device".format(device))


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            # nn.Linear(1728, 1728,bias=false), nn.ReLU(), nn.Linear(1728, 10000,bias=false),nn.ReLU(),nn.Linear(10000, 1728,bias=false),nn.ReLU(), nn.Linear(1728, 1728,bias=false))
            nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10), nn.LogSoftmax())

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("models/mnist_model",
                      map_location=torch.device(device)))

loss = nn.NLLLoss()
optimize123 = torch.optim.Adam(model.parameters(), lr=0.001)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    model.train()
    l1 = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        X = X.view(X.shape[0], -1)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        l1 += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Loss: {loss:>7f}    [{current:>5f}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    # num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X = X.view(X.shape[0], -1)

            pred = model(X)
            l = loss_fn(pred, y).item()
            print(l)
            test_loss += l
        print(test_loss)



figure = plt.figure()

def view_classify(img, ps):
    ps = ps.cpu().data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()

def total_accuracy():
    correct_count, all_count = 0, 0
    for images,labels in test1:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            with torch.no_grad():
                logps = model(img)

            
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count/all_count))


def plot_and_verify():
    img, lab = next(iter(test1))
    plt.imshow(np.array(img[0]).reshape(28,28,1))
    plt.show()
    i = img[0].view(1, 784)
    with torch.no_grad():
        r = model(i)
    ps = torch.exp(r)
    print(ps.numpy())
    probab = list(ps.numpy()[0])
    print("Predicted Digit =", probab.index(max(probab)))
    view_classify(img.view(1, 28, 28), ps)
    print("Done!")

# epochs = 1
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train1, model, loss, optimize123)
# torch.save(model.state_dict(), "models/mnist_model")

plot_and_verify()

total_accuracy()