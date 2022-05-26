import torch
from torch import Tensor, nn
import torch.utils.data as torch_data
from sklearn.model_selection import train_test_split
import pandas as pd 

from data import Iris, IrisDataLoader


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # Flattens tensor to 1D
        self.flatten = nn.Flatten()

        # Takes 5 columns, has 2 hidden layers of 8 neurons, and 3 class outputs.
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8,8),
            nn.ReLU(),
            nn.Linear(8,8),
            nn.ReLU(),
            nn.Linear(8,3)
        )
        
    def forward(self, x):
        """
        Send input through NN.
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
def train(dataloader: torch_data.DataLoader, model: nn, loss_fn, optimizer):
    """
    Trains NN given some data, a loss function, and an optimizer.
    """
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader: torch_data.DataLoader, model: nn, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

data = IrisDataLoader(batch_size=10)
epochs = 1000

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(data.train_dataloader, model, loss_fn, optimizer)
    test(data.test_dataloader, model, loss_fn)
print("Done!")

"""
Notes from playing around with the hyperparameters:
1 Hidden Layer: Too few neurons to do anything with. Appx. 40% accuracy.
    - Activation functions did not change the outcome
    - Adding Epochs did not help
    - Reducing the batch size marginally helped.
    - Reducing the learning rate marginally helped.
4 Hidden Layers: Seemed to have too many neurons. Adding more epochs did not help, and tweaking the learning rate by factors of ten didn't have an effect.
2 Hidden Layers: Seems to be the goldilocks setting. It has enough complexity to explain the dataset, but not so much that it requires a large dataset. 

Reasonable next steps (If I didn't get 100% accuracy):
- Early stopping
- Hyperparameter optimization
"""