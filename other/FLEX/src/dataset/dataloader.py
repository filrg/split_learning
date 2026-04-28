import random

import torch
import torchvision
from collections import defaultdict
from tqdm import tqdm

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from transformers import BertTokenizer
from datasets import load_dataset

from src.dataset.AGNEWS import AGNEWS_DATASET
from src.dataset.SPEECHCOMMANDS import SpeechCommandsDataset

def AGNEWS(batch_size=None, distribution=None, train=True):
    dataset = load_dataset(
        'ag_news',
        download_mode='reuse_dataset_if_exists',
        cache_dir='./hf_cache'
    )
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    if train:
        train_data = dataset['train']
        train_target_counts = {k: v for k, v in enumerate(distribution)}
        train_by_class = defaultdict(list)
        for text, label in zip(train_data['text'], train_data['label']):
            train_by_class[label].append((text, label))

        train_texts, train_labels = [], []
        for label, count in train_target_counts.items():
            samples = random.sample(train_by_class[label], count)
            train_texts.extend([t for t, _ in samples])
            train_labels.extend([l for _, l in samples])
        print("Train samples:", len(train_texts), {l: train_labels.count(l) for l in set(train_labels)})

        train_set = AGNEWS_DATASET(train_texts, train_labels, tokenizer, max_length=128)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        return train_loader
    else:
        test_data = dataset['test']
        distribution = [500, 500, 500, 500]
        test_target_counts = {k: v for k, v in enumerate(distribution)}
        test_by_class = defaultdict(list)
        for text, label in zip(test_data['text'], test_data['label']):
            test_by_class[label].append((text, label))

        test_texts, test_labels = [], []
        for label, count in test_target_counts.items():
            samples = random.sample(test_by_class[label], count)
            test_texts.extend([t for t, _ in samples])
            test_labels.extend([l for _, l in samples])

        print("Test samples:", len(test_texts), {l: test_labels.count(l) for l in set(test_labels)})

        test_set = AGNEWS_DATASET(test_texts, test_labels, tokenizer, max_length=128)
        test_loader = DataLoader(test_set, batch_size=100, shuffle=False)
        return test_loader

def CIFAR10(batch_size=None, distribution=None, train=True):
    if train:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

        labels = train_set.targets
        label_to_indices = defaultdict(list)
        for idx, label in tqdm(enumerate(labels)):
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

def SPEECHCOMMANDS(batch_size=None, distribution=None, train=True):
    if train:
        dataset = SpeechCommandsDataset(root='./data', subset='training')

        if distribution is not None:
            from src.dataset.SPEECHCOMMANDS import CLASSES
            label_to_indices = defaultdict(list)
            for idx, (audio_path, label_name) in enumerate(dataset.samples):
                label_idx = CLASSES.index(label_name)
                label_to_indices[label_idx].append(idx)

            selected_indices = []
            for label, count in enumerate(distribution):
                if count > 0 and label in label_to_indices:
                    available = label_to_indices[label]
                    selected_indices.extend(random.sample(available, min(count, len(available))))

            print(f"[DEBUG] Selected {len(selected_indices)} samples after distribution filter")
            subset = torch.utils.data.Subset(dataset, selected_indices)
            train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return train_loader
    else:
        dataset = SpeechCommandsDataset(root='./data', subset='testing')
        test_loader = DataLoader(dataset, batch_size=20, shuffle=False)
        return test_loader

def data_loader(data_name=None, batch_size=None, distribution=None, train=True):
    if data_name == 'AGNEWS':
        data = AGNEWS(batch_size, distribution, train)
    elif data_name == 'SPEECHCOMMANDS':
        data = SPEECHCOMMANDS(batch_size, distribution, train)
    else:
        data = CIFAR10(batch_size, distribution, train)

    return data