# client.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import flwr as fl
from torch.utils.data import DataLoader, Subset
from model import CNN, get_parameters, set_parameters

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, train_loader, test_loader, device):
        self.cid = cid
        self.device = device
        self.model = CNN().to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
        # Set model parameters sent by server
        self.set_parameters(parameters)
        epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", 0.01))
        optimizer = optim.SGD(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

        # Return updated parameters and the number of examples used for training
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += self.criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        loss /= len(self.test_loader.dataset)
        accuracy = correct / len(self.test_loader.dataset)
        # Flower expects (loss, num_examples, metrics)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

def partition_dataset(dataset, num_clients=5, cid=0):
    # simple equal partition by indices
    n = len(dataset)
    per_client = n // num_clients
    start = cid * per_client
    end = start + per_client if cid != num_clients - 1 else n
    indices = list(range(start, end))
    return Subset(dataset, indices)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True, help="client id (0..4)")
    parser.add_argument("--server_address", type=str, default="localhost:8080")
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--local_epochs", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use MNIST for simplicity
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    testset  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_subset = partition_dataset(trainset, num_clients=args.num_clients, cid=args.cid)
    test_subset  = partition_dataset(testset, num_clients=args.num_clients, cid=args.cid)

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_subset,  batch_size=args.batch_size, shuffle=False)

    # Create Flower client and start
    client = FlowerClient(args.cid, train_loader, test_loader, device)

    # When start_numpy_client returns, the process stops
    fl.client.start_numpy_client(server_address=args.server_address, client=client, grpc_max_message_length=1024*1024*200)

if __name__ == "__main__":
    main()
