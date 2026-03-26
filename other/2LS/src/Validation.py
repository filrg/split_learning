import numpy as np
import math
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from src.dataset.dataloader import data_loader
from src.model import *

def test(model_name, data_name, state_dict_full, logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use centralized data_loader
    try:
        test_loader = data_loader(data_name=data_name, train=False)
    except ValueError as e:
        logger.log_error(str(e))
        return False

    # Dynamic model class lookup
    if model_name in ['Bert', 'BERT']:
        if data_name == 'AGNEWS':
            klass = Bert_AGNEWS
        else:
            klass = BERT_EMOTION
    elif model_name == 'KWT':
        klass = KWT_SPEECHCOMMANDS
    else:
        # globals() might not contain all models if they are only in src.model
        # but we did 'from src.model import *'
        klass_name = f"{model_name}_{data_name}"
        klass = globals().get(klass_name)

    if klass is None:
        logger.log_error(f"Model class for {model_name} and {data_name} not found.")
        return False

    # Initialize full model
    model = klass()

    model.load_state_dict(state_dict_full)
    model = model.to(device)
    
    # evaluation mode
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    
    report = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy)
    
    print(report)

    if np.isnan(test_loss) or math.isnan(test_loss) or abs(test_loss) > 10e5:
        return False
    else:
        logger.log_info(report)

    return True
