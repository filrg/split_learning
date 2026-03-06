import random

import torch
import torchvision
from collections import defaultdict
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from transformers import BertTokenizer
from datasets import load_dataset

from src.dataset.EMOTION import EMOTIONDataset
from src.dataset.EMOTION import load_train_EMOTION
from src.dataset.EMOTION import load_test_EMOTION

def EMOTION(batch_size=None, distribution=None, train=True):
    dataset = load_dataset(
        'ag_news',
        download_mode='reuse_dataset_if_exists',
        cache_dir='./hf_cache'
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    if train:
        train_texts, train_labels = load_train_EMOTION(dataset, distribution)
        train_set = EMOTIONDataset(train_texts, train_labels, tokenizer, max_length=128)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        return train_loader
    else:
        test_texts, test_label = load_test_EMOTION(2000, dataset)
        test_set = EMOTIONDataset(test_texts, test_label, tokenizer, max_length=128)
        test_loader = DataLoader(test_set, batch_size=100, shuffle=False)
        return test_loader

def CIFAR10(batch_size=None, distribution=None, train = True):
    if train:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform= transform_train)

        label_to_indices = defaultdict(list)
        for idx, (_, label) in tqdm(enumerate(train_set)):
            label_to_indices[int(label)].append(idx)

        selected_indices = []
        for label, count in enumerate(distribution):
            selected_indices.extend(random.sample(label_to_indices[label], count))
        subset = torch.utils.data.Subset(train_set, selected_indices)

        train_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)

        return train_loader
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=1)
        return test_loader

def MNIST(batch_size=None, distribution=None, train = True):
    if train:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True,
                                                    transform=transform_train)
        label_to_indices = defaultdict(list)
        for idx, (_, label) in tqdm(enumerate(train_set)):
            label_to_indices[int(label)].append(idx)

        selected_indices = []
        for label, count in enumerate(distribution):
            selected_indices.extend(random.sample(label_to_indices[label], count))
        subset = torch.utils.data.Subset(train_set, selected_indices)
        train_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)

        return train_loader
    else:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
        return test_loader

def data_loader(data_name=None, batch_size=None, distribution=None, train=True):
    if data_name == 'EMOTION':
        data = EMOTION(batch_size, distribution, train)
    elif data_name == 'MNIST':
        data = MNIST(batch_size, distribution, train)
    else:
        data = CIFAR10(batch_size, distribution, train)

    return data
