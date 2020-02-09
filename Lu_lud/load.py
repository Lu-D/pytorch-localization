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
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

import warnings

warnings.filterwarnings("ignore")
plt.ion()


class PhoneDataset(Dataset):

    def __init__(self, file, root, mode='/train', transform=None, test=False):
        self.data = pd.read_csv(file, sep=" ", header=None)
        if test:
            self.data.columns = ["names"]
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
        new_h, new_w = self.output_size, self.output_size
        img = transform.resize(image, (new_h, new_w))
        return {'image': img, 'coordinates': coordinates}


class Normalize(object):
    def __init__(self, inplace=False):
        self.mean = (0.5692824, 0.55365936, 0.5400631)
        self.std = (0.1325967, 0.1339596, 0.14305606)
        self.inplace = inplace

    def __call__(self, sample):
        image, coordinates = sample['image'], sample['coordinates']
        return {'image': TF.normalize(image, self.mean, self.std, self.inplace), 'coordinates': coordinates,
                'original': image}


class ToTensor(object):

    def __call__(self, sample):
        dtype = torch.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        image, coordinates = sample['image'], sample['coordinates']
        image = image.transpose((2, 0, 1))
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
    def __init__(self, p=0.2, brightness=(0.5, 1.755), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.2, 0.2)):
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


def calc_acc(input, output):
    sum = 0.0
    for i in range(input.shape[0]):
        sum += torch.sqrt((input[i][0] - output[i][0]) ** 2 + (input[i][1] - output[i][1]) ** 2)
    return sum / input.shape[0]
