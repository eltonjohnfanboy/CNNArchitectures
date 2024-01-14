# Import the libraries
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def train(model, device, train_loader, optimizer, epoch, criterion):
    
    model.train()
    loss_total = 0
    loop = tqdm(train_loader, leave = True)

    for batch_idx, (data, target) in enumerate(loop):
        
        data = data.to(device)
        target = target.to(device)
        
        out_clf1, out_clf2, out = model(data)
        
        loss = criterion(out, target) + 0.3 * criterion(out_clf1, target) + 0.3 * criterion(out_clf2, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total += loss.item()

    return loss_total/batch_idx        