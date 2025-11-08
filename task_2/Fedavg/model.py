# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # for MNIST (1 channel)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_parameters(model: nn.Module):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model: nn.Module, parameters):
    # parameters: list of NumPy ndarrays in the same order as state_dict
    state_dict = model.state_dict()
    new_state = OrderedDict()
    for (k, _), val in zip(state_dict.items(), parameters):
        new_state[k] = torch.tensor(val)
    model.load_state_dict(new_state)
