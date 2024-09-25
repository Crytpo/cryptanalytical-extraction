import os
import argparse
import datetime
import json
import copy
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import keras

from torcheval.metrics import MulticlassAccuracy
from training_loop import TrainingLoop, SimpleTrainingStep
from training_loop.callbacks import EarlyStopping

from torch.utils.tensorboard import SummaryWriter

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


def create_dataloader(args, batch_size, shuffle=True, num_workers=0):
    """
    Creates a data loader for the MNIST dataset.

    Args:
        batch_size (int): The batch size for the data loader.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        num_workers (int, optional): The number of worker threads to use. Defaults to 0.

    Returns:
        torch.utils.data.DataLoader: The data loader for the MNIST dataset.
    """

    if args.dataset == 'mnist':
        # Define transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Load the MNIST dataset
        trainset = datasets.MNIST(
            root='/data',
            train=True,
            download=True,
            transform=transform
        )

        testset = datasets.MNIST(
            root='/data',
            train=False,
            download=True,
            transform=transform
        )

    elif args.dataset == 'cifar10':
        # Define transformations
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, padding=4),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])


        # Load the MNIST dataset
        trainset = datasets.CIFAR10(
            root='/data',
            train=True,
            download=True,
            transform=train_transform
        )

        testset = datasets.CIFAR10(
            root='/data',
            train=False,
            download=True,
            transform=test_transform
        ) 

    # Create data loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def calculate_prediction(model, dataloader, device='cuda'):
    model.eval()  # Set the model to evaluation mode
    
    all_output = []
    all_targets = []
    with torch.no_grad():  # Disable gradient calculation for efficiency
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            all_output.append(outputs.data.to('cpu'))
            all_targets.append(target.data.to('cpu'))
    
    return torch.vstack(all_output), torch.concatenate(all_targets)


def create_model(args):
    """
    Creates a model based on the provided arguments.
    """
    class FlexibleNet(nn.Module):
        def __init__(self, args):
            super(FlexibleNet, self).__init__()

            # Define input channels based on dataset
            if args.dataset == 'mnist':
                self.img_size = 28
                self.in_channels = 1
            elif args.dataset == 'cifar10':
                self.img_size = 32
                self.in_channels = 3
            else:
                raise NotImplementedError(f"Dataset {args.dataset} not supported")

            layers = []
            activation = nn.ReLU()  # Assuming ReLU activation for each layer

            layer_fn = nn.Linear

            if len(args.layers) > 0: # Custom Network
                print("custom layer")
                for i, layer in enumerate(args.layers):
                    if layer['layer_type'] == 'dense':
                        layer_fn = nn.Linear

                        if i == 0:
                            layers.append(layer_fn(self.img_size**2*self.in_channels, layer['hidden_size']))
                        else:
                            layers.append(layer_fn(args.layers[i-1]['hidden_size'], layer['hidden_size']))

                    elif layer['layer_type'] == 'conv2d':
                        layer_fn = nn.Conv2d
                        if i == 0:
                            layers.append(layer_fn(self.img_size**2*self.in_channels, layer['hidden_size'], kernel_size=layer['kernel_size'], padding='valid')) 
                        else:
                            layers.append(layer_fn(args.layers[i-1]['hidden_size'], layer['hidden_size'], kernel_size=layer['kernel_size'], padding='valid')) 


                    elif layer['layer_type'] == 'avgpooling':
                        layer_fn = nn.AvgPool2d
                        layers.append(layer_fn(args.layers[i]['kernel_size'], padding=1))

                    if i == len(args.layers)-1:
                        activation = nn.Softmax(dim=-1) 

                    layers.append(activation)
            else:   # Foerster's implementation 
                # Define layer type and activation function
                if args.layer_type == 'dense':
                    layer_fn = nn.Linear
                elif args.layer_type == 'conv2d':
                    layer_fn = nn.Conv2d 
                else:
                    raise NotImplementedError(f"Layer type {args.layer_type} not supported")
                
                # First layer
                if args.layer_type == 'dense':
                    layers.append(layer_fn(self.img_size**2*self.in_channels, args.hidden_size))
                elif args.layer_type == 'conv2d':
                    layers.append(layer_fn(self.in_channels, args.hidden_size, kernel_size=args.kernel_size, padding='valid')) 
                else:
                    raise NotImplementedError(f"Layer type {args.layer_type} not supported")

                layers.append(activation)

                # Intermediate layers
                for _ in range(1, args.layer_number):
                    if args.layer_type == 'dense':
                        layers.append(layer_fn(args.hidden_size, args.hidden_size))
                    elif args.layer_type == 'conv2d':
                        layers.append(layer_fn(args.hidden_size, args.hidden_size, kernel_size=args.kernel_size, padding='valid'))
                    layers.append(activation)

                # Output layer
                if args.lastactivation == 'softmax':
                    num_classes = 10
                    expected_input_dim = args.falttodense_size if args.flattodense else args.hidden_size
                else:
                    num_classes = 1 # regression problem
                    expected_input_dim = args.falttodense_size if args.flattodense else self.img_size**2*args.hidden_size
                layers.append(nn.Linear(expected_input_dim, num_classes))

                if args.lastactivation == 'softmax':
                    layers.append(nn.Softmax(dim=1))  # Softmax activation for classification

            self.model = nn.Sequential(*layers)

        def forward(self, x):
            x = x.view(-1, self.img_size**2*self.in_channels)  # Flatten the input tensor
            x = self.model(x)
            return x
   
    model = FlexibleNet(args)
    print(model)
    return model


def mae(predicted, target):
    return torch.mean(torch.abs(predicted - target)).item()


def train_model(args, model, train_loader, test_loader, epochs, device):
    """
    Trains the model on the provided data loaders.
    """
  
    # writer = SummaryWriter(log_dir)
  
    model.to(device)

    loop = TrainingLoop(
        model,
        step=SimpleTrainingStep(
            optimizer_fn=lambda params: Adam(params, lr=0.0001),
            loss=torch.nn.CrossEntropyLoss(),
            metrics=('accuracy', MulticlassAccuracy(num_classes=10)),
        ),
        device=device,
    )
    fitted = loop.fit(
        train_loader,
        test_loader,
        epochs=args.epochs,
        callbacks=[
            EarlyStopping(monitor='val_loss', mode='min', patience=20),
        ],
    )

    # Predict the test set
    predictions_tr, y_train = calculate_prediction(model, train_loader)
    predictions, y_test  = calculate_prediction(model, test_loader)

    if  args.lastactivation == 'softmax' or args.layers[-1]['activation'] == 'softmax':
        # Round predictions to the nearest integer
        rounded_predictions_tr = np.argmax(predictions_tr, axis=1)
        rounded_predictions    = np.argmax(predictions, axis=1)
    else:
        # Round predictions to the nearest integer
        rounded_predictions_tr = np.round(predictions_tr).astype(int).flatten()
        rounded_predictions    = np.round(predictions).astype(int).flatten()

    # Calculate accuracy
    accuracy_tr = (rounded_predictions_tr == y_train).float().mean()
    accuracy    = (rounded_predictions == y_test).float().mean()

    print(f"Accuracy: {accuracy_tr * 100:.2f}%, {accuracy * 100:.2f}%")

    log_dir = f"models/logs/pt/{args.dataset}/{args.layer_type}/{args.hidden_size}x{args.layer_number}/{args.lastactivation}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    print('Finished Training. Save data ...')
    print('Log dir', log_dir)

    args.train_acc = accuracy_tr.item()
    args.test_acc = accuracy.item()
    args_dict = copy.deepcopy(vars(args))
    with open(os.path.join(log_dir, 'args.json'), 'w') as fp:
        json.dump(args_dict, fp, indent=4)

    # writer.close()
    # checkpoint = {'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}
    checkpoint = {'state_dict': model.state_dict(),'accuracy': {"train": args.train_acc, "test": args.test_acc}}
    torch.save(checkpoint, os.path.join(log_dir, 'model.pt'))

    print(model)


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
    parser.add_argument('--epochs', default=100, type=int, help='')
    parser.add_argument('--lr', default=0.001, type=float, help='')
    parser.add_argument('--bs', default=64, type=int, help='')
    parser.add_argument('--seed', default=42, type=int, help='')

    parser.add_argument('--load_json', default="", type=str, help='')

    args = parser.parse_args()

    if len(args.load_json) > 0:
        with open(args.load_json, "r") as f:
            json_data = json.load(f)
            args.__dict__.update(json_data)

    #args.epochs = 1
    print(args)

    model = create_model(args)
    train_loader, test_loader = create_dataloader(args, batch_size=args.bs)

    train_model(args, model, train_loader, test_loader, epochs=args.epochs, device="cuda")
