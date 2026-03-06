import torch
import torch.nn as nn
import time
import numpy as np
from src.model.Bert_EMOTION import Bert
from src.model.VGG16_CIFAR10 import VGG16_CIFAR10
from src.model.VGG16_MNIST import VGG16_MNIST
from src.dataset.dataloader import data_loader
from tqdm import tqdm
import pickle
import pika
import uuid
import json

import os
from peft import LoraConfig, TaskType, get_peft_model

import argparse

parser = argparse.ArgumentParser(description="Profiling Processing")
parser.add_argument('--layer_id', type=int, required=True, help='ID of layer, start from 1')
parser.add_argument('--model', type=str, required=True, help='Model name')
parser.add_argument('--data', type=str, required=True, help='Data name')
parser.add_argument('--rounds', type=int, required=False, help='Number round of profiling')
parser.add_argument('--size', type=int, required=False, help='Batch size')

args = parser.parse_args()

client_id = uuid.uuid4()

def profiling_bert(layer_id, batch_size=4, rounds=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = data_loader('EMOTION', batch_size, [50, 50, 50, 50], True)
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        r=4, lora_alpha=8, lora_dropout=0.1,
        bias="none",
        target_modules=["query", "key", "value", "dense"]
    )
    criterion = nn.CrossEntropyLoss()
    exec_time = []
    for i in range(1, 12):
        print(f'Num blocks: {i}')
        if layer_id == 1:
            exec_t = 0
            model = Bert(layer_id=1, n_block=i)
            model = get_peft_model(model, peft_config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.01)
            model = model.to(device)

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
                    exec_t += ((time.time() - start) * 1e9)
                    break
            exec_time.append(exec_t / rounds)
        else:
            exec_t = 0
            model = Bert(layer_id=2, n_block=i)
            model = get_peft_model(model, peft_config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.01)
            model = model.to(device)
            for _ in tqdm(range(rounds)):
                model.train()
                optimizer.zero_grad()
                batch = next(iter(train_loader))
                labels = batch['labels'].to(device)

                x = torch.randn(batch_size, 128, 768).to(device)
                start = time.time()
                output = model(input_ids=x)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                exec_t += ((time.time() - start) * 1e9)

            exec_time.append(exec_t / rounds)

    exec_time.reverse()
    x = torch.randn(batch_size, 128, 768)
    data_size = x.nelement() * x.element_size()

    total_time = 0
    model = Bert(layer_id=0, n_block=12)
    model = get_peft_model(model, peft_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.01)
    model = model.to(device)
    for _ in tqdm(range(rounds)):
        model.train()
        optimizer.zero_grad()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            start = time.time()
            output = model(input_ids=input_ids)
            loss =  criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_time += ((time.time() - start) * 1e9)
            break

    total_time = (total_time / rounds)

    info = {"execute training time": exec_time,
            "list of data size": data_size,
            "training speed": round(float(32 / (total_time * 1e-9)), 2)}
    return info

def profiling_vgg(data_name, size=32, rounds=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if data_name == 'CIFAR10':
        model = VGG16_CIFAR10()
    else :
        model = VGG16_MNIST()
    test_loader =  data_loader(data_name, size, train=False)

    data_size = []
    forward_time = []

    full_model = []
    for sub_model in nn.Sequential(*nn.ModuleList(model.children())):
        full_model.append(sub_model)

    weight_backward = []
    for name, module in model.named_children():
        if 'Conv2d' in str(module):
            weight_backward.append(3)
        else:
            weight_backward.append(2)

    for i in tqdm(range(rounds)):
        img = None
        for (img, target) in test_loader:
            img = img.to(device)
            break

        times = []
        for sub_model in full_model:
            sub_model = sub_model.to(device)
            sub_model.train()
            start = time.time()
            img = sub_model(img)
            end = time.time()
            if i == 0:
                data_size.append(img.nelement() * img.element_size())
            times.append((end - start) * 1e9)
        forward_time.append(times)

    forward_time = np.array(forward_time)
    forward_time = np.average(forward_time, axis=0)
    backward_time = forward_time * np.array(weight_backward)
    forward_time = forward_time.tolist()
    backward_time = backward_time.tolist()
    exe_time = [a + b for a, b in zip(forward_time, backward_time)]
    info = {"execute training time": exe_time,
            "total time": int(sum(exe_time)),
            "list of data size": data_size,
            "training speed": round(float(32 / (sum(exe_time) * 1e-9)), 2)}
    return info

def profiling(model_name=None, data_name=None, layer_id=0, size=4, rounds=100):
    if os.path.exists("profiling.json"):
        os.remove("profiling.json")
    if model_name == 'bert':
        info = profiling_bert(layer_id, batch_size=size, rounds=rounds)
    else:
        info = profiling_vgg(data_name, size=size, rounds=rounds)

    return info


def network(channel, rounds = 100, id_client = None):
    speed_all = []
    size_data = []
    for i in range(1, 10):
        size_data.append(i * 10**6)
    queue_name = f"test_network_{id_client}"
    channel.queue_declare(queue=queue_name, durable=True)

    for size in size_data:
        message = size * '1'
        avg_time = 0.0
        for _ in tqdm(range(rounds)):
            time_stamp = time.time()
            channel.basic_publish(exchange='',
                                  routing_key=queue_name,
                                  body=pickle.dumps(message),
                                  properties=pika.BasicProperties(
                                      expiration='10',  # TTL = 10 milliseconds
                                      delivery_mode=1  # non-persistent (optional)
                                    )
                                  )
            avg_time += ((time.time() - time_stamp) * 10 ** 9)
        avg_time = avg_time / rounds
        speed = size / avg_time
        speed_all.append(speed)
    channel.queue_delete(queue=queue_name)
    speed = sum(speed_all) / len(speed_all)
    speed =  round(speed, 4)

    return speed

data = profiling(args.model, args.data, args.layer_id, args.size, args.rounds)

credentials = pika.PlainCredentials('dai', 'dai')
connection = pika.BlockingConnection(pika.ConnectionParameters('192.168.101.92', 5672, f'/', credentials))
channel_mq = connection.channel()

net = network(channel_mq, 100, client_id)
data['network'] =  net
with open('profiling.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print(f'End profiling')
