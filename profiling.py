import torch
import time
from src.model.BERT_AGNEWS import BERT_AGNEWS
from src.model.VGG16_CIFAR10 import VGG16_CIFAR10
from src.model.KWT_SPEECHCOMMANDS import KWT_SPEECHCOMMANDS
from tqdm import tqdm
import pickle
import pika
import uuid
import json

import argparse

parser = argparse.ArgumentParser(description="Profiling Processing")
parser.add_argument('--model', type=str, required=True, help='Model name')
parser.add_argument('--size', type=int, required=False, help='Batch size')

args = parser.parse_args()

client_id = uuid.uuid4()

def register_hooks(model, profile: dict):

    def pre_hook(name):
        def hook(module, input):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            profile[name] = {}
            profile[name]['start'] = time.time()
        return hook

    def post_hook(name):
        def hook(module, input, output):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            profile[name]['end'] = time.time()
            profile[name]['time'] = profile[name]['end'] - profile[name]['start']
            profile[name]['size'] = output.nelement() * output.element_size()

        return hook

    for name, layer in model.named_children():
        layer.register_forward_pre_hook(pre_hook(name))
        layer.register_forward_hook(post_hook(name))

def profiling(model_name=None, size=4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    info = {"exe_time": [], "size_data": []}
    profile = {}

    if model_name == 'VGG16':
        model = VGG16_CIFAR10().to(device)
        register_hooks(model, profile)
        x = torch.randn(size, 3, 32, 32).to(device)
    elif model_name == 'BERT':
        model = BERT_AGNEWS().to(device)
        register_hooks(model, profile)
        x = torch.randn(size, 128).to(device)
    else:
        model = KWT_SPEECHCOMMANDS().to(device)
        register_hooks(model, profile)
        x = torch.randn(size, 40, 98).to(device)

    # warm up
    for _ in range(30):
        _ = model(x)

    _ = model(x)
    for k, v in profile.items():
        info["exe_time"].append(round(v['time'] * 1e9 * 3, 2))
        info["size_data"].append(v['size'])

    speed = round(float(size / (sum(info["exe_time"]) * 1e-9)), 2)
    info["speed"] = speed
    return info

def network(channel, id_client = None):
    speed_all = []
    size_data = []
    for i in range(1, 10):
        size_data.append(i * 10**6)
    queue_name = f"test_network_{id_client}"
    channel.queue_declare(queue=queue_name, durable=True)

    for size in size_data:
        message = size * '1'
        avg_time = 0.0
        for _ in tqdm(range(50)):
            time_stamp = time.time()
            channel.basic_publish(exchange='',
                                  routing_key=queue_name,
                                  body=pickle.dumps(message),
                                  properties=pika.BasicProperties(
                                      expiration='10',
                                      delivery_mode=1
                                    )
                                  )
            avg_time += ((time.time() - time_stamp) * 10 ** 9)
        avg_time = avg_time / 50
        speed = size / avg_time
        speed_all.append(speed)
    channel.queue_delete(queue=queue_name)
    speed = sum(speed_all) / len(speed_all)
    speed =  round(speed, 4)

    return speed

data = profiling(args.model, args.size)

credentials = pika.PlainCredentials('admin', 'admin')
connection = pika.BlockingConnection(pika.ConnectionParameters('127.0.0.1', 5672, f'/', credentials))
channel_mq = connection.channel()

net = network(channel_mq, client_id)
data['network'] =  net
with open('profiling.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print(f'End profiling')
