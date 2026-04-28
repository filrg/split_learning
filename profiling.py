import torch
import torch.optim as optim
import time
from src.model.BERT_AGNEWS import BERT_AGNEWS
from src.model.VGG16_CIFAR10 import VGG16_CIFAR10
from src.model.KWT_SPEECHCOMMANDS import KWT_SPEECHCOMMANDS
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
parser.add_argument('--round', type=int, required=False, help='Number round of profiling')
parser.add_argument('--size', type=int, required=False, help='Batch size')

args = parser.parse_args()

client_id = uuid.uuid4()

size_VGG16 = [[32, 3, 32, 32], [32, 64, 32, 32], [32, 64, 32, 32], [32, 64, 32, 32], [32, 64, 32, 32], [32, 64, 32, 32], [32, 64, 32, 32], [32, 64, 16, 16], [32, 128, 16, 16], [32, 128, 16, 16], [32, 128, 16, 16], [32, 128, 16, 16], [32, 128, 16, 16], [32, 128, 16, 16], [32, 128, 8, 8], [32, 256, 8, 8], [32, 256, 8, 8], [32, 256, 8, 8], [32, 256, 8, 8], [32, 256, 8, 8], [32, 256, 8, 8], [32, 256, 8, 8], [32, 256, 8, 8], [32, 256, 8, 8], [32, 256, 4, 4], [32, 512, 4, 4], [32, 512, 4, 4], [32, 512, 4, 4], [32, 512, 4, 4], [32, 512, 4, 4], [32, 512, 4, 4], [32, 512, 4, 4], [32, 512, 4, 4], [32, 512, 4, 4], [32, 512, 2, 2], [32, 512, 2, 2], [32, 512, 2, 2], [32, 512, 2, 2], [32, 512, 2, 2], [32, 512, 2, 2], [32, 512, 2, 2], [32, 512, 2, 2], [32, 512, 2, 2], [32, 512, 2, 2], [32, 512, 1, 1], [32, 512], [32, 512], [32, 4096], [32, 4096], [32, 4096], [32, 4096], [32, 4096]]
size_BERT = [[2, 128], [2, 128, 768], [2, 128, 768], [2, 128, 768], [2, 128, 768], [2, 128, 768], [2, 128, 768], [2, 128, 768], [2, 128, 768], [2, 128, 768], [2, 128, 768], [2, 128, 768], [2, 128, 768], [2, 128, 768], [2, 768]]
size_KWT = [[32, 40, 98], [32, 98, 64], [32, 99, 64], [32, 99, 64], [32, 99, 64], [32, 99, 64], [32, 99, 64], [32, 99, 64], [32, 99, 64], [32, 99, 64], [32, 99, 64], [32, 99, 64], [32, 99, 64], [32, 99, 64], [32, 99, 64], [32, 99, 64], [32, 64]]

def profiling_BERT(layer_id, size=2, rounds=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        r=4, lora_alpha=8, lora_dropout=0.1,
        bias="none",
        target_modules=["query", "key", "value", "dense"]
    )

    time_exe = []
    data_size = []
    total_time = 0

    for i in tqdm(range(1,15)):
        if layer_id == 1:
            model = BERT_AGNEWS(end_layer=i).to(device)
            input_ids = torch.ones([size, 128], dtype=torch.long).to(device)
            if i > 3:
                model = get_peft_model(model, peft_config)
        else:
            model = BERT_AGNEWS(start_layer=i).to(device)
            input_ids = torch.ones(size_BERT[i]).to(device)
            if i < 14:
                model = get_peft_model(model, peft_config)

        exec_t = 0
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.01)
        model = model.to(device)
        model.train()

        for r in range(rounds):
            start = time.time()
            optimizer.zero_grad()
            out_put = model(input_ids=input_ids)
            if r == 0:
                data_size.append(out_put.nelement() * out_put.element_size())
            loss = out_put.mean()
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
            if r >= 10:
                exec_t += ((time.time() - start) * 1e9)
        time_exe.append(exec_t / (rounds - 10))

    model = BERT_AGNEWS()
    model = get_peft_model(model, peft_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.01)
    model = model.to(device)
    model.train()
    input_ids = torch.ones([size, 128], dtype=torch.long).to(device)
    for rou in tqdm(range(rounds)):
        optimizer.zero_grad()
        start = time.time()
        out_put = model(input_ids=input_ids)
        loss =  out_put.mean()
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        if rou >= 10:
            total_time += ((time.time() - start) * 1e9)

    total_time = (total_time / (rounds - 10))

    info = {"execute training time": time_exe,
            "list of data size": data_size,
            "training speed": round(float(4 / (total_time * 1e-9)), 2)}
    return info


def profiling_VGG16(layer_id=1, size=32, rounds=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    time_exe = []
    data_size = []
    total_time = 0

    for i in tqdm(range(1, 52)):
        if layer_id == 1:
            model = VGG16_CIFAR10(end_layer=i).to(device)
            img = torch.ones([size, 3, 32, 32]).to(device)
        else:
            model = VGG16_CIFAR10(start_layer=i).to(device)
            img = torch.ones(size_VGG16[i]).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
        model.train()
        exec_t = 0

        for r in range(rounds):
            start =  time.time()
            optimizer.zero_grad()
            out_put = model(img)
            if r == 0:
                data_size.append(out_put.nelement() * out_put.element_size())
            loss = out_put.mean()
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
            if r >= 10:
                exec_t += ((time.time() - start) * 1e9)

        time_exe.append(exec_t / (rounds - 10))

    model = VGG16_CIFAR10().to(device)
    img = torch.ones([size, 3, 32, 32]).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    for rou in tqdm(range(rounds)):
        model.train()
        optimizer.zero_grad()
        start = time.time()
        out_put = model(img)
        loss = out_put.mean()
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        if rou >= 10:
            total_time += ((time.time() - start) * 1e9)

    total_time = total_time / (rounds - 10)
    info = {"execute training time": time_exe,
            "list of data size": data_size,
            "training speed": round(float(32 / (total_time * 1e-9)), 2)}

    return info

def profiling_KWT(layer_id = 1, size=32, rounds=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    size_data = []
    time_exe = []
    total_time = 0

    for i in tqdm(range(1, 17)):
        if layer_id == 1:
            model = KWT_SPEECHCOMMANDS(end_layer=i).to(device)
            x = torch.ones([size, 40, 98]).to(device)
        else:
            model = KWT_SPEECHCOMMANDS(start_layer=i).to(device)
            x = torch.ones(size_KWT[i]).to(device)
        model.to(device)

        exec_t = 0
        optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=1e-4)
        model.train()

        for r in range(rounds):
            start = time.time()
            optimizer.zero_grad()
            out_put = model(x)
            if r == 0:
                size_data.append(out_put.nelement() * out_put.element_size())
            loss = out_put.mean()
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()
            if r >= 10:
                exec_t += ((time.time() - start) * 1e9)
        time_exe.append(exec_t / (rounds - 10))

    model = KWT_SPEECHCOMMANDS().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.00001, weight_decay=1e-4)
    model.train()
    x = torch.ones([size, 40, 98]).to(device)
    for rou in tqdm(range(rounds)):
        optimizer.zero_grad()
        start = time.time()
        out_put = model(x)
        loss = out_put.mean()
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        if rou >= 10:
            total_time += ((time.time() - start) * 1e9)

    total_time = (total_time / (rounds - 10))
    info = {"execute training time": time_exe,
            "list of data size": size_data,
            "training speed": round(float(8 / (total_time * 1e-9)), 2)}

    return info


def profiling(model_name=None, layer_id=1, size=4, rounds=100):
    if os.path.exists("profiling.json"):
        os.remove("profiling.json")
    if model_name == 'BERT':
        info = profiling_BERT(layer_id, size=size, rounds=rounds)
    elif model_name == 'KWT':
        info = profiling_KWT(layer_id, size=size, rounds=rounds)
    else:
        info = profiling_VGG16(layer_id, size=size, rounds=rounds)

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

data = profiling(args.model, args.layer_id, args.size, args.round)

credentials = pika.PlainCredentials('dai', 'dai')
connection = pika.BlockingConnection(pika.ConnectionParameters('192.168.101.92', 5672, f'/', credentials))
channel_mq = connection.channel()

net = network(channel_mq, 100, client_id)
data['network'] =  net
with open('profiling.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print(f'End profiling')
