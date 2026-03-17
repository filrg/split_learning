import torch
import torch.nn as nn
from tqdm import tqdm

from src.model.Bert_AGNEWS import Bert_AGNEWS
from src.dataset.dataloader import data_loader

def val_Bert(data_name, state_dict_full, logger):
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = data_loader(data_name=data_name,train=False)
    model = Bert_AGNEWS()
    model = model.to(device)
    model.load_state_dict(state_dict_full)

    model.eval()
    correct, total, total_loss = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        avg_loss = total_loss / len(test_loader)

    print(f"Test Loss: {avg_loss:.2f}; Test Acc: {acc:.2f}")

    logger.log_info(f"Test Loss: {avg_loss:.2f}; Test Acc: {acc:.2f}")








