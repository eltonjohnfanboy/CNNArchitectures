import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

# Class for the inception block. We will stack these to form the complete GoogleNet
class InceptionBlock(nn.Module):

    def __init__(self, input_dim, output1x1, output3x3nin, output3x3, output5x5nin, output5x5, output1x1mp):
        super().__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(input_dim, output1x1, kernel_size = 1),
            nn.BatchNorm2d(output1x1),
            nn.ReLU()
        )
        
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(input_dim, output3x3nin, kernel_size = 1),
            nn.BatchNorm2d(output3x3nin),
            nn.ReLU(),
            nn.Conv2d(output3x3nin, output3x3, kernel_size = 3, padding = 'same'),
            nn.BatchNorm2d(output3x3),
            nn.ReLU()
        )

        self.conv5x5 = nn.Sequential(
            nn.Conv2d(input_dim, output5x5nin, kernel_size = 1),
            nn.BatchNorm2d(output5x5nin),
            nn.ReLU(),
            nn.Conv2d(output5x5nin, output5x5, kernel_size = 5, padding = 'same'),
            nn.BatchNorm2d(output5x5),
            nn.ReLU()
        )

        self.conv1x1withmaxpooling = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
            nn.Conv2d(input_dim, output1x1mp, kernel_size = 1),
            nn.BatchNorm2d(output1x1mp),
            nn.ReLU()
        )
    
    def forward(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)
        out1x1mp = self.conv1x1withmaxpooling(x)

        # Concatenate the result of each "branch"
        return torch.cat([out1x1, out3x3, out5x5, out1x1mp], dim = 1)


# Auxiliary classifiers that let the gradient flow to the lower layers (also work as regularizers)
class AuxiliaryClassifier(nn.Module):

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size = 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        

# The whole GoogLeNet architecture
class GoogLeNet(nn.Module):

    def __init__(self, input_dim, num_classes, aux_classifiers = True):
        super().__init__()
        self.aux_classifiers = aux_classifiers
        self.phase1 = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
            nn.Conv2d(64, 192, kernel_size = 3, stride = 1, padding = 'same'),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size = 7, stride = 1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)       

        if self.aux_classifiers:
            self.aux1 = AuxiliaryClassifier(512, num_classes)
            self.aux2 = AuxiliaryClassifier(528, num_classes)
        else:
            self.aux1 = self.aux2 = None

    def forward(self, x):
        out = self.phase1(x)
        out = self.inception3a(out)
        out = self.inception3b(out)
        out = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)(out)
        out = self.inception4a(out)

        if self.training and self.aux_classifiers:
            out_clf1 = self.aux1(out)

        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)

        if self.training and self.aux_classifiers:
            out_clf2 = self.aux2(out)

        out = self.inception4e(out)
        out = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)(out)
        out = self.inception5a(out)
        out = self.inception5b(out)
        out = self.avgpool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.dropout(out)
        out = self.fc(out)

        if self.training and self.aux_classifiers:
            return out_clf1, out_clf2, out

        return out