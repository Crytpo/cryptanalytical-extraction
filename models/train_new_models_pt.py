import os
from icecream import ic
import argparse
import datetime
import json
import copy
import numpy as np
import pandas as pd 

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from training_loop import TrainingLoop, SimpleTrainingStep
from training_loop.callbacks import EarlyStopping, TensorBoardLogger, ModelCheckpoint
import torchmetrics.functional as tmf
from torcheval.metrics import MulticlassAccuracy
from torchmetrics import MeanAbsoluteError

from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

# import logging
# torch._logging.set_logs(
#     all=logging.DEBUG, 
#     output_code=True, 
#     graph_code=True,
#     sym_node=True,
#     )

import torch.autograd.profiler as profiler
# https://pytorch.org/docs/stable/bottleneck.html

dataset_config = {
    'mnist':   {'img_size': 28, 'in_channels': 1, 'classes': 10},
    'cifar10': {'img_size': 32, 'in_channels': 3, 'classes': 10}
}


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
        transform = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            # transforms.Normalize((0.5,), (0.5,))
        ]
        transform = transforms.Compose(transform)

        if args.lastactivation == 'softmax':
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
        else:   
            class CustomMNISTDataset(Dataset):
                def __init__(self, root, train=True, transform=None, target_type=torch.float32):
                    self.dataset = datasets.MNIST(root=root, train=train, transform=transform)
                    self.target_type = target_type

                def __len__(self):
                    return len(self.dataset)

                def __getitem__(self, idx):
                    img, target = self.dataset[idx]
                    target = torch.tensor(target, dtype=self.target_type)  # Convert target to desired dtype
                    return img, target     
                
            trainset = CustomMNISTDataset("/data", train=True, transform=transform)
            testset  = CustomMNISTDataset("/data", train=False, transform=transform)


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


def predict(args, model, data_loader, num_classes=10, task='MULTICLASS'):
    model.eval()
    predictions = []
    targets = []

    for data, target in data_loader:
        data = data.to(args.device)
        target = target.to(args.device)

        # Forward pass
        with torch.no_grad():
            output = model(data)
        
        # Accumulate predictions and targets
        predictions.append(output)
        targets.append(target)

    # Concatenate predictions and targets
    predictions = torch.cat(predictions, dim=0).cpu()
    targets = torch.cat(targets, dim=0).cpu()

    if args.lastactivation == 'softmax':
        # Calculate accuracy
        accuracy = tmf.accuracy(predictions, targets, task=task, num_classes=num_classes).item()*100
        print(f"Prediction accuracy: {accuracy:.2f}%")
    else:
        # regression
        predictions = predictions.round().int()
        targets = targets.int()
        accuracy = MeanAbsoluteError()(predictions, targets).item()
        print(f"Prediction accuracy: {accuracy:.4f} MAE")
        accuracy2 = (predictions == targets).float().mean().item()*100
        print(f"Prediction accuracy: {accuracy2:.2f} %")
        accuracy = accuracy2

    return accuracy


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
    if len(args.layers) > 0: # Custom Network
        # Define input channels based on dataset
        class CustomNN(nn.Module):
            def __init__(self, args):
                super(CustomNN, self).__init__()

                self.args = args
                self.img_size = dataset_config[args.dataset]['img_size']
                self.in_channels = dataset_config[args.dataset]['in_channels']

                layers = []
                activation = nn.ReLU() # Assuming ReLU activation for each layer
                layer_fn = nn.Linear
                ic("custom network")
                for i, layer in enumerate(args.layers):
                    if layer['layer_type'] == 'dense':
                        layer_fn = nn.Linear

                        if i == 0:
                            layers.append(layer_fn(self.img_size**2*self.in_channels, layer['hidden_size']))
                        else:
                            layers.append(layer_fn(args.layers[i-1]['hidden_size'], layer['hidden_size']))
                        
                        if not i == len(args.layers) - 1:
                            layers.append(activation)

                    elif layer['layer_type'] == 'conv2d':
                        layer_fn = nn.Conv2d
                        if i == 0:
                            layers.append(layer_fn(self.in_channels, layer['hidden_size'], kernel_size=layer['kernel_size'], padding=1)) 
                        else:
                            layers.append(layer_fn(args.layers[i-1]['hidden_size'], layer['hidden_size'], kernel_size=layer['kernel_size'], padding=1)) 

                        if not i == len(args.layers) - 1:
                            layers.append(activation)

                    elif layer['layer_type'] == 'avgpooling':
                        layer_fn = nn.AvgPool2d
                        layers.append(layer_fn(args.layers[i]['kernel_size'], args.layers[i]['stride'],  padding=1))

                    elif layer['layer_type'] == 'flatten':
                        layers.append(nn.Flatten())

                self.model = nn.Sequential(*layers)

            def forward(self, x):
                if self.args.layers[0]['layer_type'] == 'dense':
                    x = torch.flatten(x, start_dim=1)

                x = self.model(x)

                if not self.args.lastactivation == 'softmax':
                    x = x.squeeze()
                    
                return x
            
        model = CustomNN(args)
            
    else:
        if args.layer_type == 'dense':
            class MLP(nn.Module):
                def __init__(self, args, hidden_size, layer_number):
                    super(MLP, self).__init__()
                    self.img_size = dataset_config[args.dataset]['img_size']
                    self.in_channels = dataset_config[args.dataset]['in_channels']
                    self.layers = nn.ModuleList()
                    self.layers.append(nn.Linear(self.img_size**2*self.in_channels, hidden_size))
                    for _ in range(1, layer_number):
                        self.layers.append(nn.Linear(hidden_size, hidden_size))

                    if args.lastactivation == 'linear':
                        self.output = nn.Linear(hidden_size, 1)  # Output layer for regression-like approach
                    else:
                        self.output = nn.Linear(hidden_size, dataset_config[args.dataset]['classes']) 
                    
                def forward(self, x):
                    x = torch.flatten(x, start_dim=1)
                    for layer in self.layers:
                        x = torch.relu(layer(x))
                    
                    if not self.args.lastactivation == 'softmax':
                        out = x.squeeze()
                    
                    return out
                
            model = MLP(args, args.hidden_size, args.layer_number)
    
        elif args.layer_type == 'conv2d':
            class MLP(nn.Module):
                def __init__(self, args, hidden_size, layer_number):
                    super(MLP, self).__init__()
                    self.img_size = dataset_config[args.dataset]['img_size']
                    self.in_channels = dataset_config[args.dataset]['in_channels']
                    self.layers = nn.ModuleList()
                    self.layers.append(nn.Conv2d(self.in_channels, hidden_size, kernel_size=args.kernel_size, padding=1))
                    for _ in range(1, layer_number):
                        self.layers.append(nn.Conv2d(hidden_size, hidden_size, kernel_size=args.kernel_size, padding=1))

                    # if args.flattodense:
                    # expected_input_dim = args.falttodense_size if args.flattodense else args.hidden_size*self.img_size**2
                    self.flatten = nn.Flatten()
                    
                    expected_input_dim = hidden_size*self.img_size**2
                    if args.flattodense:
                        expected_output_dim = args.falttodense_size if args.falttodense_size > 0 else hidden_size
                        self.flattodense_layer = nn.Linear(expected_input_dim, expected_output_dim)
                        expected_input_dim = expected_output_dim

                    if args.lastactivation == 'linear':
                        self.output = nn.Linear(expected_input_dim, 1)  # Output layer for regression-like approach
                    else:
                        self.output = nn.Linear(expected_input_dim, dataset_config[args.dataset]['classes'])

                def forward(self, x):
                    for layer in self.layers:
                        x = torch.relu(layer(x))
                    
                    x = self.flatten(x) 

                    if args.flattodense:
                        x = torch.relu(self.flattodense_layer(x))

                    if args.lastactivation == 'linear':
                        out = self.output(x).squeeze()  # Output layer for regression-like approach
                    else:
                        out = self.output(x)
                    
                    return out
                
            model = MLP(args, args.hidden_size, args.layer_number)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            # nn.init.xavier_uniform_(m.weight)
            # m.bias.data.fill_(0.0)
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            m.bias.data.fill_(0.01)

    model.apply(init_weights)

    print(model)
    summary(model)
    summary(model, torch.randn(args.bs, model.in_channels, model.img_size, model.img_size))

    return model


def train_model(args, model, train_loader, test_loader):
    """
    Trains the model on the provided data loaders.
    """
    log_dir = f"models/logs/pt/{args.dataset}/{args.layer_type}/{args.hidden_size}x{args.layer_number}/{args.lastactivation}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    ic('Log dir', log_dir)

    tbCallBack = TensorBoardLogger(log_dir, update_freq=1, update_freq_unit="epoch")
    ckptCallBack = ModelCheckpoint(
        os.path.join(log_dir, "best_model.pt"),
        save_weights_only=True,
        save_best_only=True,
        monitor="loss",
        mode="min",
    )
    # predict(args, model, train_loader) # debug

    if args.lastactivation == 'softmax':
        loop = TrainingLoop(
            model,
            step=SimpleTrainingStep(
                optimizer_fn=lambda params: Adam(params, lr=args.lr),
                loss=nn.CrossEntropyLoss(),
                metrics=('accuracy', MulticlassAccuracy(num_classes=dataset_config[args.dataset]['classes'])),
            ),
            device=args.device,
        )
    else:
        loop = TrainingLoop(
            model,
            step=SimpleTrainingStep(
                optimizer_fn=lambda params: Adam(params, lr=args.lr),
                loss=nn.MSELoss(),  # Use Mean Squared Error loss for regression
                metrics=('mean_absolute_error', MeanAbsoluteError()),  # Use Mean Absolute Error metric
            ),
            device=args.device,
        )

    tr_per, te_per = loop.fit(
        train_loader,
        test_loader,
        epochs=args.epochs,
        callbacks=[
            EarlyStopping(monitor='val_loss', mode='min', patience=15),
            tbCallBack,
            ckptCallBack
        ],
        verbose=2
    )

    accuracy_tr = predict(args, model, train_loader, num_classes=dataset_config[args.dataset]['classes'])
    accuracy    = predict(args, model, test_loader,  num_classes=dataset_config[args.dataset]['classes'])
    
    ic(f"Accuracy: {accuracy_tr:.2f}%, {accuracy:.2f}%")
    ic('Finished Training. Save data args ...')

    args.train_acc = accuracy_tr
    args.test_acc = accuracy
    args_dict = copy.deepcopy(vars(args))
    with open(os.path.join(log_dir, 'args.json'), 'w') as fp:
        json.dump(args_dict, fp, indent=4)

    # writer.close()
    # checkpoint = {'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}
    checkpoint = {'state_dict': model.state_dict(),'accuracy': {"train": args.train_acc, "test": args.test_acc}}
    torch.save(checkpoint, os.path.join(log_dir, 'model.pt'))
    
    tr_per.to_parquet(os.path.join(log_dir, "train_perfomance.parquet"))
    te_per.to_parquet(os.path.join(log_dir, "test_perfomance.parquet"))

    ic(model)
    ic("finished!")
    ic(f"Data saved under '{log_dir}'")


if __name__ == '__main__':
    ic("if gpu available, this code uses it otherwise it uses CPU!")
    ic(torch.cuda.device_count())
    ic("--------------------------------------------")

    parser = argparse.ArgumentParser(description='Create model')
    parser.add_argument('--dataset', default="mnist", choices=['mnist', 'cifar10'], type=str, help='')
    parser.add_argument('--layer_type', default="dense", choices=['dense', 'conv2d'], type=str, help='')
    parser.add_argument('--layers', default=[], type=list, help='')
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
    parser.add_argument('--device', default="cuda", type=str, help='')

    parser.add_argument('--load_json', default="", type=str, help='')

    args = parser.parse_args()

    if len(args.load_json) > 0:
        with open(args.load_json, "r") as f:
            json_data = json.load(f)
            args.__dict__.update(json_data)

    ic(args)

    # Set the seed for the CPU
    torch.manual_seed(args.seed)

    # Set the seed for the GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = create_model(args)
    train_loader, test_loader = create_dataloader(args, batch_size=args.bs)

    train_model(args, model, train_loader, test_loader)
