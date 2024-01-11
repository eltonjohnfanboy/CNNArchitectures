# Import the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import dataset as ds
from train import train
from test import test
from model import VGGnet

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import the datasets and create the dataloaders
train_ds = ds.train_dataset
test_ds = ds.test_dataset

train_loader = torch.utils.data.DataLoader(train_ds, batch_size = 256, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size = 256, shuffle = False)

# Define the model + weight initialitzation 
# (used Xavier uniform, even though in the paper they use random initialization or weights of previously trained shallow networks)
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight.data)

input_dim = 3
num_classes = 10
model = VGGnet(input_dim, num_classes).to(device) # (input_dim, num_classes)
model.apply(initialize_weights)

# Set hyperparameters
EPOCHS = 100
lr = 0.01

# Optimitzer and loss
optimizer = optim.SGD(model.parameters(), lr=lr, momentum = 0.9) # we use the same parameters as the paper
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