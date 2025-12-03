import torch
import torch.nn as nn
import time
import numpy as np
import torchvision
import torchvision.transforms as transforms
import src.Model
from tqdm import tqdm

def profiling(rounds = 100, batch_size = 32, device = "cpu"):

    model = src.Model.VGG16().to(device)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    data_size = []
    forward_time = []

    full_model = []
    for sub_model in nn.Sequential(*nn.ModuleList(model.children())):
        full_model.append(sub_model)

    weight_backward = []
    for name, module in model.named_children():
        # print(f"Name: {name}, Module: {module}")
        if 'Conv2d' in str(module):
            weight_backward.append(3)
        else:
            weight_backward.append(2)

    for i in tqdm(range(rounds)):
        data = None
        for (data, target) in test_loader:
            data = data.to(device)
            break

        times = []
        for sub_model in full_model:
            sub_model.train()
            start = time.time()
            data = sub_model(data)
            end = time.time()
            if i == 0:
                data_size.append(data.nelement() * data.element_size())
            times.append((end-start) * 1e9)
        forward_time.append(times)

    forward_time = np.array(forward_time)
    forward_time = np.average(forward_time, axis=0)
    backward_time = forward_time * np.array(weight_backward)
    forward_time = forward_time.tolist()
    backward_time = backward_time.tolist()
    exe_time = [a + b for a, b in zip(forward_time, backward_time)]
    data = {"execute training time": exe_time,
            "total time": int(sum(exe_time)),
            "list of data size": data_size,
            "training speed": round(float(batch_size / (sum(exe_time) * 1e-9 )), 2)}
    return data
