# Import the libraries
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def test(model, device, test_loader, optimizer, epoch, criterion): # crec que podem posar-ho tot en config
    
    loss_test = 0
    
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        loop = tqdm(test_loader, leave = True)
        
        for batch_idx, (data, target) in enumerate(loop):
            
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            
            loss = criterion(output, target)
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            loss_test += loss.item()
            
        accur = 100 * correct / total
            
        return loss_test/batch_idx, accur




