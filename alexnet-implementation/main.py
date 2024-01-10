# Import the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import dataset as ds
from train import *
from test import *
from model import AlexNet

# Device configuration -> to use the GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import the datasets and create the dataloaders
train_ds = ds.train_dataset
test_ds = ds.test_dataset

train_loader = torch.utils.data.DataLoader(train_ds, batch_size = 256, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size = 256, shuffle = False)

# Define the model + weight initialitzation
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')

input_dim = 3
num_classes = 10
model = AlexNet(input_dim, num_classes).to(device) # (input_dim, num_classes)
model.apply(initialize_weights)

# Set hyperparameters
EPOCHS = 100
lr = 0.01

# Optimitzer and loss
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = 0.005)
criterion = nn.CrossEntropyLoss()

# Lists we will use to plot the loss function
loss_history_train = []
loss_history_test = []

for e in range(EPOCHS):
    loss_train = train(model, device, train_loader, optimizer, e, criterion)
    loss_test, accur_test = test(model, device, test_loader, criterion)
                
    print('Epoch: {} \tTrainLoss: {:.6f}\tValidationLoss: {:.6f}\tAccuracy validation: {:.6f}'.format(
        e, 
        loss_train,
        loss_test,
        accur_test
        ))
    
    loss_history_train.append(loss_train)
    loss_history_test.append(loss_test)
    plt.plot(range(len(loss_history_train)), loss_history_train)
    plt.plot(range(len(loss_history_test)), loss_history_test)
    plt.show()
    
    # Save the model weights every epoch
    PATH = r"trained_models/"
    torch.save(model.state_dict(), PATH + "model1.pth")
