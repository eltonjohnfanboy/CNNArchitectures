# Import the libraries
import numpy as np
import torch.nn as nn
from tqdm import tqdm

def train(model, device, train_loader, optimizer, epoch, criterion):
    
    model.train()
    loss_total = 0
    loop = tqdm(train_loader, leave = True)

    for batch_idx, (data, target) in enumerate(loop):
        
        data = data.to(device)
        target = target.to(device)
        
        out = model(data)
        
        loss = criterion(out, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total += loss.item()

    return loss_total/batch_idx        