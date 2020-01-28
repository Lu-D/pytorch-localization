# Author: Daiwei (David) Lu
# A fully custom dataloader for the cellphone dataset

import os
import torch
import pandas as pd
from PIL import Image
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils
import torchvision.transforms.functional as TF
import random


import warnings
warnings.filterwarnings("ignore")
plt.ion()


class PhoneDataset(Dataset):

    def __init__(self, file, root, mode='/train', transform=None, test=False):
        self.data = pd.read_csv(file, sep=" ", header=None)
        if test:
            self.data.columns=["names"]
        else:
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
        dtype = torch.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        image, coordinates = sample['image'], sample['coordinates']
        image = image.transpose((2,0,1))
        return {'image': torch.from_numpy(image).type(dtype),
                'coordinates': torch.from_numpy(coordinates).type(dtype)}

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, coordinates = sample['image'], sample['coordinates']
        if random.random() < self.p:
            image *= 255
            image = Image.fromarray(np.uint8(image))
            image = TF.hflip(image)
            image = np.array(image)
            image = np.float32(image) / 255.
            coordinates[0] = 1.0 - coordinates[0]
        return {'image': image, 'coordinates': coordinates}

class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, coordinates = sample['image'], sample['coordinates']
        if random.random() < self.p:
            image *= 255.
            image = Image.fromarray(np.uint8(image))
            image = TF.vflip(image)
            image = np.array(image)
            image = np.float32(image) / 255.
            coordinates[1] = 1.0 - coordinates[1]
        return {'image': image, 'coordinates': coordinates}

class RandomColorJitter(object):
    def __init__(self, p=0.2, brightness=(0.5, 1.755), contrast=(0.5,1.5), saturation=(0.5,1.5), hue=(-0.2,0.2)):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, sample):
        image, coordinates = sample['image'], sample['coordinates']
        if random.random() < self.p:
            image *= 255.
            image = Image.fromarray(np.uint8(image))
            modifications = []

            brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
            modifications.append(transforms.Lambda(lambda img: TF.adjust_brightness(image, brightness_factor)))

            contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
            modifications.append(transforms.Lambda(lambda img: TF.adjust_contrast(image, contrast_factor)))

            saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
            modifications.append(transforms.Lambda(lambda img: TF.adjust_saturation(image, saturation_factor)))

            hue_factor = random.uniform(self.hue[0], self.hue[1])
            modifications.append(transforms.Lambda(lambda img: TF.adjust_hue(image, hue_factor)))

            random.shuffle(modifications)
            modification = transforms.Compose(modifications)
            image = modification(image)

            image = np.array(image)
            image = np.float32(image) / 255.
        return {'image': image, 'coordinates': coordinates}


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