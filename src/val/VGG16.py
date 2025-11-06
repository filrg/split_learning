from tqdm import tqdm

from src.dataset.dataloader import data_loader

from src.model import *

def val_VGG16(data_name, state_dict_full, logger):
    test_loader = data_loader(data_name=data_name, train=False)

    klass = globals()[f'VGG16_{data_name}']
    model = klass()
    model.load_state_dict(state_dict_full)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            outputs = model(images)

            _, predicts = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicts == labels).sum().item()

    acc = 100 * correct / total

    print('Test set:Accuracy: {}/{} ({:.2f}%)\n'.format(
     correct, total, acc))
    logger.log_info('Test set:Accuracy: {}/{} ({:.2f}%)\n'.format(
     correct, total, acc))