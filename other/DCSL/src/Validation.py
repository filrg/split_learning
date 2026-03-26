import numpy as np
import math
from tqdm import tqdm

from src.dataset.dataloader import data_loader
from src.model import *

def test(model_name, data_name, state_dict_full, logger, server_connection=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = data_loader(data_name=data_name, train=False)

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
