from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import time
import copy
from load import *


import warnings
warnings.filterwarnings("ignore")
plt.ion()

def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 9001.
    best_acc = 1.
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = ['x', 'y']
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
            running_corrects = 0.
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
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += calc_acc(labels, outputs)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.6f}'.format(
                phase, epoch_loss))
            print('{} Acc: {:.6f}'.format(
                phase, epoch_acc
            ))

            # deep copy the model
            if phase == 'val' and epoch_acc < best_acc:
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:6f}'.format(best_loss))
    print('Best val Acc: {:6f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, dataloaders, device):
    was_training = model.training
    model.eval()
    fig = plt.figure()

    with torch.no_grad():
        for i, batch in enumerate(dataloaders['val']):
            inputs, labels = batch['image'], batch['coordinates']
            inputs = inputs.float().cuda().to(device)
            print('Label:', batch['coordinates'].data)
            batch['coordinates'].data = model(inputs).data
            plt.figure()
            batch_show(batch)
            plt.axis('off')
            plt.ioff()
            plt.show()

        model.train(mode=was_training)

image_datasets = {'train': PhoneDataset('labels/train.txt',
                       '',
                       mode='train',
                       transform=transforms.Compose([
                           Rescale(256),
                           RandomVerticalFlip(0.5),
                           RandomHorizontalFlip(0.5),
                           RandomColorJitter(0.9),
                           ToTensor()
                       ])),
                'val': PhoneDataset('labels/val.txt',
                       '',
                       mode='validation',
                       transform=transforms.Compose([
                           Rescale(256),
                           RandomVerticalFlip(0.1),
                           RandomHorizontalFlip(0.1),
                           RandomColorJitter(0.1),
                           ToTensor()
                       ]))}

def main():
        dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32,
                                             shuffle=True),
                       'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=4,
                                             shuffle=True)}

        device = torch.device("cuda")

        model = models.vgg11(pretrained=True)
        model.classifier = nn.Sequential(*[model.classifier[i] for i in range(3)])
        model.classifier = nn.Sequential(nn.Linear(25088, 64),
                                         nn.Dropout(0.5),
                                         nn.Linear(64, 2),)
        model = model.to(device)

        criterion = nn.MSELoss()


        optimizer_conv = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.5 every 20 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=20, gamma=0.5)

        print(model)
        model = train_model(model, criterion, optimizer_conv,
                                 exp_lr_scheduler, dataloaders, device, num_epochs=25)

        visualize_model(model, dataloaders, device)
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    main()