import json
import pika
import uuid
import argparse
import yaml
import os

import torch

import src.Log
from src.RpcClient import RpcClient

parser = argparse.ArgumentParser(description="Split learning framework")
parser.add_argument('--layer_id', type=int, required=True, help='ID of layer, start from 1')
parser.add_argument('--device', type=str, required=False, help='Device of client')
parser.add_argument('--cluster', type=int, required=False, help='ID cluster by device')

args = parser.parse_args()

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

client_id = uuid.uuid4()
address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]
virtual_host = config["rabbit"]["virtual-host"]

device = None
if args.device is None:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: {torch.cuda.get_device_name(device)}")
    else:
        device = "cpu"
        print(f"Using device: CPU")
else:
    device = args.device
    print(f"Using device: {device}")

credentials = pika.PlainCredentials(username, password)
connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials))
channel = connection.channel()

if args.cluster is None:
    cluster = -1
else:
    cluster = args.cluster

if __name__ == "__main__":
    src.Log.print_with_color("[>>>] Client sending registration message to server...", "red")
    if os.path.exists("profiling.json"):
        src.Log.print_with_color(f"Exists profiling.json.", 'green')
        with open("profiling.json","r", encoding='utf-8') as file:
            loaded_data = json.load(file)

        performance = loaded_data["training speed"]
        exe_time = loaded_data["execute training time"]
        net = loaded_data["network"]
        size_data = loaded_data["list of data size"]

        data = {"action": "REGISTER", "client_id": client_id, "layer_id": args.layer_id, "performance": performance ,"cluster": cluster, "exe_time": exe_time, "net": net, "size_data": size_data, "message": "Hello from Client!"}
        client = RpcClient(client_id, args.layer_id, channel, device)
        client.send_to_server(data)
        client.wait_response()
    else:
        src.Log.print_with_color("[>>>] Profiling file is not existing, break", "yellow")


