import torch
import torch.nn as nn
import torch.optim as optim

import time
import numpy as np
import torchvision
import torchvision.transforms as transforms
from src.model.VGG16_CIFAR10 import VGG16_CIFAR10
from src.dataset.dataloader import data_loader
from tqdm import tqdm

def profiling(model_name='VGG16', data_name='CIFAR10', epoch=100, batch_size=32, device='cpu'):
    if model_name == 'VGG16':
        model = VGG16_CIFAR10().to(device)
    else:
        model = VGG16_CIFAR10().to(device)

    distribution = [320,320,320,320,320,320,320,320,320,320]
    train_loader = data_loader(data_name,batch_size,distribution,True)
    train_time = []

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.05)
    criterion = nn.CrossEntropyLoss()

    for i in range(epoch):
        print(f'Epoch {i}:')
        model.train()
        start = time.time()
        for (data, label) in tqdm(train_loader):
            data = data.to(device)
            label = label.to(device)

            output = model(data)
            loss = criterion(output, label)
            loss.backward()

        end = time.time()
        train_time.append(end - start)

    return train_time


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: {torch.cuda.get_device_name(device)}")
    else:
        device = "cpu"
        print(f"Using device: CPU")

    t = profiling('VGG16', 'CIFAR10', 3, 32, device)
    print(t)

