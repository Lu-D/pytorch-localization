import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class TestFile(Dataset):

    def __init__(self, file, transform=None):
        self.file = file
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        sample = io.imread(self.file)
        if self.transform:
            sample = self.transform(sample)
        return sample

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        new_h, new_w = self.output_size,self.output_size
        img = transform.resize(sample, (new_h, new_w))
        return img

class ToTensor(object):

    def __call__(self, sample):
        dtype = torch.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        image = sample.transpose((2,0,1))
        return torch.from_numpy(image).type(dtype)

def show_dot(image, coordinates):
    plt.imshow(image)
    plt.scatter(image.shape[1]*coordinates[0][0], image.shape[0]*coordinates[0][1], marker = '.', c='r')
    plt.pause(0.001)