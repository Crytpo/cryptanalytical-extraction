import argparse
import datetime
import json
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

def str2bool(v):
    """
    Converts string to bool type; enables command line arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_mnist_dataloader(batch_size, shuffle=True, num_workers=0):
    """
    Creates a data loader for the MNIST dataset.

    Args:
        batch_size (int): The batch size for the data loader.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        num_workers (int, optional): The number of worker threads to use. Defaults to 0.

    Returns:
        torch.utils.data.DataLoader: The data loader for the MNIST dataset.
    """

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the MNIST dataset
    trainset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    testset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Create data loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def create_model(args):
    """
    Creates a model based on the provided arguments.
    """

    # Define input channels based on dataset
    if args.dataset == 'mnist':
        in_channels = 1
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported")

    # Define layer type and activation function
    if args.layer_type == 'dense':
        layer_fn = nn.Linear
    elif args.layer_type == 'conv2d':
        layer_fn = nn.Conv2d 
    else:
        raise NotImplementedError(f"Layer type {args.layer_type} not supported")

    activation = nn.ReLU()

    # Create model layers
    layers = []

    # First layer
    if args.layer_type == 'conv2d':
        layers.append(layer_fn(in_channels, args.hidden_size, kernel_size=args.kernel_size, padding=1))  # Adjust padding as needed
    elif args.layer_type == 'dense':
        layers.append(layer_fn(in_channels * args.hidden_size * args.hidden_size, args.hidden_size))
    else:
        raise NotImplementedError(f"Layer type {args.layer_type} not supported")

    layers.append(activation)  # Assuming ReLU activation for each layer

    for _ in range(1, args.layer_number):
        layers.append(layer_fn(args.hidden_size, args.hidden_size, kernel_size=args.kernel_size, padding='valid'))
        layers.append(activation)

    # Flatten and dense layers if specified
    layers.append(nn.Flatten())

    if args.flattodense:
        if args.falttodense_size < 0:
            args.falttodense_size = args.hidden_size
        layers.append(nn.Linear(28*28*args.hidden_size, args.falttodense_size))
        layers.append(nn.ReLU())  # Assuming ReLU activation

    # Output layer
    if args.lastactivation == 'softmax':
        num_classes = 10
    else:
        num_classes = 1
        
    expected_input_dim = args.falttodense_size if args.flattodense else 28*28*args.hidden_size
    print(f"Expected input dimension for the last linear layer: {expected_input_dim}")

    layers.append(nn.Linear(expected_input_dim, num_classes))

    # Construct the model using nn.Sequential
    model = nn.Sequential(*layers)
    
    print(model)
    return model


def train_model(model, train_loader, test_loader, epochs, device):
    """
    Trains the model on the provided data loaders.
    """
    model.to(device)

    # Define the optimizer
    learning_rate = 0.001  # Adjust the learning rate as needed
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define loss function based on last activation
    if args.lastactivation == 'softmax':
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(epochs):
        
        running_loss = 0.0
        model.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            breakpoint()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

        # Testing loop
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                predicted = torch.argmax(output.data, dim=1) if args.lastactivation == 'softmax' else model.out_func(output).round().long()
                total += target.size(0)
                correct += (predicted == target).sum().item()

            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1} - Accuracy: {accuracy:.2f}%")

    print('Finished Training')


if __name__ == '__main__':
    print("if gpu available, this code uses it otherwise it uses CPU!")
    print(torch.cuda.device_count())
    print("--------------------------------------------")

    parser = argparse.ArgumentParser(description='Create model')
    parser.add_argument('--dataset', default="mnist", choices=['mnist', 'cifar10'], type=str, help='')
    parser.add_argument('--layer_type', default="dense", choices=['dense', 'conv2d'], type=str, help='')
    parser.add_argument('--hidden_size', default=8, type=int, help='')
    parser.add_argument('--layer_number', default=1, type=int, help='')
    parser.add_argument('--kernel_size', default=3, type=int, help='')
    parser.add_argument('--flattodense', default=True, type=str2bool, help='')
    parser.add_argument('--falttodense_size', default=-1, type=int, help='')
    parser.add_argument('--lastactivation', default="linear", type=str, help='')
    parser.add_argument('--epochs', default=1, type=int, help='')
    parser.add_argument('--bs', default=64, type=int, help='')
    parser.add_argument('--seed', default=42, type=int, help='')

    parser.add_argument('--load_json', default="", type=str, help='')

    args = parser.parse_args()

    if len(args.load_json) > 0:
        with open(args.load_json, "r") as f:
            json_data = json.load(f)
            args.__dict__.update(json_data)

    print(args)

    model = create_model(args)
    train_loader, test_loader = create_mnist_dataloader(batch_size=args.bs)
    train_model(model, train_loader, test_loader, epochs=args.epochs, device="cuda")