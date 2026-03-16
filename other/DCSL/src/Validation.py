
import numpy as np
import math
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from src.model import *

def test(model_name, data_name, state_dict_full, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if data_name == "MNIST":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    elif data_name == "CIFAR10":
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif data_name == "SPEECHCOMMANDS":
        from src.dataset.SPEECHCOMMANDS import SpeechCommandsDataset
        testset = SpeechCommandsDataset(root='./data', subset='testing')
    elif data_name == "EMOTION":
        from datasets import load_dataset
        from transformers import BertTokenizer
        from src.dataset.EMOTION import EMOTIONDataset, load_test_EMOTION
        dataset = load_dataset('ag_news', download_mode='reuse_dataset_if_exists', cache_dir='./hf_cache')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        test_texts, test_labels = load_test_EMOTION(1000, dataset)
        testset = EMOTIONDataset(test_texts, test_labels, tokenizer, max_length=128)
    else:
        raise ValueError(f"Data name '{data_name}' is not valid.")

    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()

    klass = globals()[f'{model_name}_{data_name}']

    if klass is None:
        raise ValueError(f"Class '{model_name}' does not exist.")

    model = klass()

    model.load_state_dict(state_dict_full)
    model = model.to(device)
    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * target.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    test_loss /= total
    accuracy = 100.0 * correct / total
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, total, accuracy))

    if np.isnan(test_loss) or math.isnan(test_loss) or abs(test_loss) > 10e5:
        return False
    else:
        logger.log_info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, total, accuracy))

    return True
