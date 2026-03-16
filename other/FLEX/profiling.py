import torch
import time
import numpy as np
from src.model.Bert_AGNEWS import Bert_AGNEWS
from src.model.VGG16_CIFAR10 import VGG16_CIFAR10
from src.dataset.dataloader import data_loader
from tqdm import tqdm
import json

import os
from peft import LoraConfig, TaskType, get_peft_model

import argparse

parser = argparse.ArgumentParser(description="Profiling Processing")
parser.add_argument('--model', type=str, required=True, help='Model name')
parser.add_argument('--data', type=str, required=True, help='Data name')
parser.add_argument('--rounds', type=int, required=False, help='Number round of profiling')
parser.add_argument('--size', type=int, required=False, help='Batch size')

args = parser.parse_args()

def profiling_bert(batch_size=4, rounds=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = data_loader('EMOTION', batch_size, [50, 50, 50, 50], True)
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        r=4, lora_alpha=8, lora_dropout=0.1,
        bias="none",
        target_modules=["query", "key", "value", "dense"]
    )
    exec_time = 0
    model = Bert_AGNEWS(layer_id=0, n_block=12)
    model = get_peft_model(model, peft_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.01)
    model = model.to(device)
    model.train()
    optimizer.zero_grad()
    for _ in tqdm(range(rounds)):
        model.train()
        optimizer.zero_grad()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            start = time.time()
            output = model(input_ids=input_ids)
            loss = output.mean()
            loss.backward()
            optimizer.step()
            exec_time += ((time.time() - start) * 1e9)
            break

    exec_time = exec_time / rounds


    info = {"speed": round(float(batch_size / (exec_time * 1e-9)), 2)}
    return info

def profiling_vgg(data_name, size=32, rounds=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG16_CIFAR10()
    model = model.to(device)
    model.train()
    test_loader =  data_loader(data_name, size, train=False)
    times = []

    for _ in tqdm(range(rounds)):
        img = None
        for (img, target) in test_loader:
            img = img.to(device)
            break

        start = time.time()
        _ = model(img)
        end = time.time()
        times.append((end - start) * 1e9)

    times = np.array(times)
    times = np.average(times, axis=0)
    info = {"speed": round(float(size / (times * 1e-9)), 2)}
    return info

def profiling(model_name=None, data_name=None, size=4, rounds=100):
    if os.path.exists("profiling.json"):
        os.remove("profiling.json")
    if model_name == 'bert':
        info = profiling_bert(batch_size=size, rounds=rounds)
    else:
        info = profiling_vgg(data_name, size=size, rounds=rounds)

    return info

data = profiling(args.model, args.data, args.size, args.rounds)

with open('profiling.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print(f'End profiling')
