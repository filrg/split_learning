
import numpy as np
import math
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from src.model import *

def test(model_name, data_name, state_dict_full, logger, server_connection=None):
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
        testset_full = SpeechCommandsDataset(root='./data', subset='testing')
        indices = np.random.choice(len(testset_full), 5000, replace=False)
        testset = torch.utils.data.Subset(testset_full, indices)
    elif data_name == "AGNEWS":
        from datasets import load_dataset
        from transformers import BertTokenizer
        from src.dataset.AGNEWS import AGNEWS_DATASET, load_test_AGNEWS
        dataset = load_dataset('ag_news', download_mode='reuse_dataset_if_exists', cache_dir='./hf_cache')
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        test_texts, test_labels = load_test_AGNEWS(1000, dataset)
        testset = AGNEWS_DATASET(test_texts, test_labels, tokenizer, max_length=128)
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
        for i, batch in enumerate(tqdm(test_loader)):
            if isinstance(batch, dict) and 'input_ids' in batch:
                data = batch['input_ids'].to(device)
                target = batch['labels'].to(device)
                output = model(data, attention_mask=batch['attention_mask'].to(device))
            else:
                data, target = batch
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                
            loss = criterion(output, target)
            test_loss += loss.item() * target.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if server_connection is not None and i % 5 == 0:
                try:
                    server_connection.process_data_events(time_limit=0)
                except Exception:
                    pass

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
