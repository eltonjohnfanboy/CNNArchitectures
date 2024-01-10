# Import the libraries
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def train(model, device, train_loader, optimizer, epoch, criterion): # crec que podem posar-ho tot en config
    
    model.train()
    loss_total = 0
    loop = tqdm(train_loader, leave = True)

    for batch_idx, (data, target) in enumerate(loop):
        
        data = data.to(device)
        target = target.to(device)
        
        output = model(data)
        
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total += loss.item()

    return loss_total/batch_idx        