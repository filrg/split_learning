import torch
import torch.nn as nn

from src.model.VGG16_CIFAR10 import VGG16_CIFAR10
from tqdm import tqdm
from src.dataset.dataloader import data_loader

def val_VGG16(data_name, state_dict_full, logger):
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    test_loader = data_loader(data_name=data_name, train=False)

    model = VGG16_CIFAR10()
    model = model.to(device)
    model.load_state_dict(state_dict_full)
    model.eval()

    correct, total, total_loss = 0, 0, 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    acc = (correct / total) * 100
    avg_loss = total_loss / len(test_loader)

    print('Test set:Loss: {:.4f}; Accuracy: {}/{} ({:.2f}%)\n'.format(avg_loss,
     correct, total, acc))
    logger.log_info('Test set:Loss: {:.4f}; Accuracy: {}/{} ({:.2f}%)\n'.format(avg_loss,
     correct, total, acc))