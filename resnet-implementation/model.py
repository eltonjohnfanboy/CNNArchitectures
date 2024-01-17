import torch.nn as nn
import torch.nn.functional as F

# Define the Residual Block
class ResBlock(nn.Module):

    def __init__(self, input_dim, output_dim, stride = 1, downsample = None):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(output_dim),
        )
        self.downsample = downsample
        self.relu = nn.ReLU()

    
    def forward(self, x):
        org_input = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            org_input = self.downsample(x)
        out += org_input
        out = self.relu(out)
        return out

# ResNet architecture
class ResNet(nn.Module):

    def __init__(self, block, list_layers, num_classes):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, list_layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, list_layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, list_layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, list_layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size = 1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out