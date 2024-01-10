import numpy as np
import torch
import torch.nn as nn


class AlexNet(nn.Module):

    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(input_dim, 96, kernel_size = 11, stride = 4, padding = 0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features = 9216, out_features = 4096, bias = True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features = 4096, out_features = 4096, bias = True),
            nn.ReLU(),
            nn.Linear(in_features = 4096, out_features = num_classes, bias = True),
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = out.view(-1, out.shape[1]*out.shape[2]*out.shape[3])
        out = self.classifier(out)
        return out
    