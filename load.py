import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")
plt.ion()

def show_dot(image, coordinates, scaled=False):
    plt.imshow(image)
    plt.scatter(image.shape[1]*coordinates[0], image.shape[0]*coordinates[1], marker = '.', c='r')
    plt.pause(0.001)


class PhoneDataset(Dataset):

    def __init__(self, file, root, mode='/train', transform=None):
        self.data = pd.read_csv(file, sep=" ", header=None)
        self.data.columns = ["names", "x", "y"]
        self.root = root
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root + self.mode, self.data.names[idx])
        image = io.imread(img_name)
        coords = np.array([self.data.x[idx], self.data.y[idx]])
        sample = {'image': image, 'coordinates': coords}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, coordinates = sample['image'], sample['coordinates']
        new_h, new_w = self.output_size,self.output_size
        img = transform.resize(image, (new_h, new_w))


        return {'image': img, 'coordinates': coordinates}

class ToTensor(object):

    def __call__(self, sample):
        image, coordinates = sample['image'], sample['coordinates']
        image = image.transpose((2,0,1))
        return {'image': torch.from_numpy(image), 'coordinates': torch.from_numpy(coordinates)}


def batch_show(sample_batched):
    """Show image for a batch of samples."""
    images_batch, coordinates_batch = \
            sample_batched['image'], sample_batched['coordinates']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(coordinates_batch[i, 0].cpu().numpy()*256 + i * im_size + (i + 1) * grid_border_size,
                    coordinates_batch[i, 1].cpu().numpy()*256 + grid_border_size,
                    marker='.', c='r')
        plt.title('Batch from dataloader')

def calc_acc(input, output):
    sum = 0.0
    for i in range(input.shape[0]):
        sum += torch.sqrt((input[i][0] - output[i][0])**2 + (input[i][1] - output[i][1])**2)
    return sum / input.shape[0]