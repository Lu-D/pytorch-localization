# Author: Daiwei (David) Lu
# Find the PhoneDataset mean and std for normalization

from load import *
from torch.utils.data import DataLoader

dataset = PhoneDataset('data/labels/train.txt',
                       'data',
                       mode='/train',
                       transform=transforms.Compose([
                           Rescale(256),
                           ToTensor(),
                           Normalize()
                       ]))
loader = DataLoader(
    dataset,
    batch_size=10,
    num_workers=1,
    shuffle=False
)

pixel_mean = np.zeros(3)
pixel_std = np.zeros(3)
k = 1
for load in loader:
    imgs = load['image']
    imgs = np.array(imgs)
    print(imgs.shape)
    for i in range(imgs.shape[0]):
        image = imgs[i]
        pixels = image.reshape((-1, image.shape[2]))

        for pixel in pixels:
            diff = pixel - pixel_mean
            pixel_mean += diff / k
            pixel_std += diff * (pixel - pixel_mean)
            k += 1

pixel_std = np.sqrt(pixel_std / (k - 2))
print(pixel_mean)
print(pixel_std)
