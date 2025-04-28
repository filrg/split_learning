import torch
import torch.nn as nn
import numpy as np
import math
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from src.model import *

def test(model_name, data_name, state_dict_full, logger):
    if data_name == "MNIST":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
    elif data_name == "FASHION_MNIST":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
    elif data_name == "CIFAR10":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Data name '{data_name}' is not valid.")

    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    if 'MNIST' in data_name:
        klass = globals()[f'{model_name}_MNIST']
    else:
        klass = globals()[f'{model_name}_{data_name}']
    if klass is None:
        raise ValueError(f"Class '{model_name}' does not exist.")

    model = klass()
    if model_name != 'ViT':
         model = nn.Sequential(*nn.ModuleList(model.children()))
    model.load_state_dict(state_dict_full)
    # evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in tqdm(test_loader):
        output = model(data)
        test_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    if np.isnan(test_loss) or math.isnan(test_loss) or abs(test_loss) > 10e5:
        return False
    else:
        logger.log_info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), accuracy))

    return True
