# Pytorch Object Localization

This project is a custom trained network for my Deep Learning in Med Imaging Course at Vanderbilt

***

![Figure](/nn.svg)

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1         [-1, 64, 256, 256]           1,792
              ReLU-2         [-1, 64, 256, 256]               0
         MaxPool2d-3         [-1, 64, 128, 128]               0
            Conv2d-4        [-1, 128, 128, 128]          73,856
              ReLU-5        [-1, 128, 128, 128]               0
         MaxPool2d-6          [-1, 128, 64, 64]               0
            Conv2d-7          [-1, 256, 64, 64]         295,168
              ReLU-8          [-1, 256, 64, 64]               0
            Conv2d-9          [-1, 256, 64, 64]         590,080
             ReLU-10          [-1, 256, 64, 64]               0
        MaxPool2d-11          [-1, 256, 32, 32]               0
           Conv2d-12          [-1, 512, 32, 32]       1,180,160
             ReLU-13          [-1, 512, 32, 32]               0
           Conv2d-14          [-1, 512, 32, 32]       2,359,808
             ReLU-15          [-1, 512, 32, 32]               0
        MaxPool2d-16          [-1, 512, 16, 16]               0
           Conv2d-17          [-1, 512, 16, 16]       2,359,808
             ReLU-18          [-1, 512, 16, 16]               0
           Conv2d-19          [-1, 512, 16, 16]       2,359,808
             ReLU-20          [-1, 512, 16, 16]               0
        MaxPool2d-21            [-1, 512, 8, 8]               0
        AdaptiveAvgPool2d-22    [-1, 512, 7, 7]               0
           Linear-23                   [-1, 64]       1,605,696
          Sigmoid-24                   [-1, 64]               0
          Dropout-25                   [-1, 64]               0
           Linear-26                    [-1, 2]             130
----------------------------------------------------------------

The implementation deals with localizing a single cell phone in images.

It is built in a VGG model style with increasing convolutional layers and decreasing dimensions.


