import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.nn.modules.distance import PairwiseDistance
import time
import copy
import torch.nn.functional as F
import argparse
from load import *


import warnings
warnings.filterwarnings("ignore")
plt.ion()

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 9001
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for loader in dataloaders[phase]:
                inputs, labels = loader['image'], loader['coordinates']
                inputs = inputs.float().cuda().to(device)
                labels = labels.float().cuda().to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # print(torch.mean(torch.abs(outputs - labels.data)))
                # statistics
                # print(preds)
                # print(labels.data)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += calc_acc(labels, outputs)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            print('{} Acc: {:.4f}'.format(
                phase, epoch_acc
            ))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    fig = plt.figure()

    with torch.no_grad():
        for i, batch in enumerate(dataloaders['val']):
            inputs, labels = batch['image'], batch['coordinates']
            inputs = inputs.float().cuda().to(device)
            print('Label:', batch['coordinates'].data)
            batch['coordinates'].data = model(inputs).data
            print('Prediction:', batch['coordinates'].data)
            plt.figure()
            batch_show(batch)
            plt.axis('off')
            plt.ioff()
            plt.show()
        model.train(mode=was_training)

# def main():
#     # Training settings
#     parser = argparse.ArgumentParser(description='Training')
#     parser.add_argument('--batch-size', type=int, default=16, metavar='N',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                         help='input batch size for testing (default: 1000)')
#     parser.add_argument('--epochs', type=int, default=25, metavar='N',
#                         help='number of epochs to train (default: 14)')
#     parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
#                         help='learning rate (default: 1.0)')
#     parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
#                         help='Learning rate step gamma (default: 0.7)')
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='disables CUDA training')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')
#     parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                         help='how many batches to wait before logging training status')
#
#     parser.add_argument('--save-model', action='store_true', default=True,
#                         help='For Saving the current Model')
#     args = parser.parse_args()
#     use_cuda = not args.no_cuda and torch.cuda.is_available()
#
#     torch.manual_seed(args.seed)
#
#     device = torch.device("cuda" if use_cuda else "cpu")
#
#     kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
#     train_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('../data', train=True, download=True,
#                        transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))
#                        ])),
#         batch_size=args.batch_size, shuffle=True, **kwargs)
#     test_loader = torch.utils.data.DataLoader(
#         datasets.MNIST('../data', train=False, transform=transforms.Compose([
#                            transforms.ToTensor(),
#                            transforms.Normalize((0.1307,), (0.3081,))
#                        ])),
#         batch_size=args.test_batch_size, shuffle=True, **kwargs)
#
#     model = Net().to(device)
#     optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
#
#     scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
#     for epoch in range(1, args.epochs + 1):
#         train(args, model, device, train_loader, optimizer, epoch)
#         test(args, model, device, test_loader)
#         scheduler.step()
#
#     if args.save_model:
#         torch.save(model.state_dict(), "mnist_cnn.pt")
#
#
# if __name__ == '__main__':
#     main()

image_datasets = {'train': PhoneDataset('data/labels/train.txt',
                       'data',
                       mode='/train',
                       transform=transforms.Compose([
                           Rescale(256),
                           ToTensor()
                       ])),
                'val': PhoneDataset('data/labels/val.txt',
                       'data',
                       mode='/validation',
                       transform=transforms.Compose([
                           Rescale(256),
                           ToTensor()
                       ]))}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = ['x', 'y']

device = torch.device("cuda")

model = models.vgg11(pretrained=True)
# for param in model.parameters():
    # param.requires_grad = False
# Parameters of newly constructed modules have requires_grad=True by default
# model.features = nn.Sequential(*[model.features[i] for i in range(11)])
# model.classifier = nn.Sequential(*[model.classifier[i] for i in range(3)])
model.classifier = nn.Sequential(nn.Linear(25088, 128),
                                 nn.Dropout(0.5),
                                 nn.Linear(128, 2))
# model.classifier= nn.Sequential(nn.Linear(2, 2))
model = model.to(device)

criterion = nn.MSELoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
# optimizer_conv = optim.Adam(model.features.parameters(), lr=0.0005)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
print(model)

model = train_model(model, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)