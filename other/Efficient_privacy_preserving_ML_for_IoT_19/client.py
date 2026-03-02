import pika
import uuid
import argparse
import yaml

import torch

import src.Log
from src.RpcClient import RpcClient
from src.Scheduler import Scheduler

parser = argparse.ArgumentParser(description="Split learning framework")
parser.add_argument('--layer_id', type=int, required=True, help='ID of layer, start from 1')
parser.add_argument('--device', type=str, required=False, help='Device of client')
# add new argument
parser.add_argument('--in_cluster_id', type=int, required=False, default=0, help='In-cluster ID')
parser.add_argument('--out_cluster_id', type=int, required=False, default=0, help='Out-cluster ID')

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

in_cluster_id = args.in_cluster_id
out_cluster_id = args.out_cluster_id

if __name__ == "__main__":
    src.Log.print_with_color("[>>>] Client sending registration message to server...", "red")

    data = {"action": "REGISTER", "client_id": client_id, "layer_id": args.layer_id, 
            "in_cluster_id": in_cluster_id, "out_cluster_id": out_cluster_id, "message": "Hello from Client!"}
    scheduler = Scheduler(client_id, args.layer_id, channel, device)
    client = RpcClient(client_id, args.layer_id, channel, scheduler.train_on_device, device)
    client.send_to_server(data)
    client.wait_response()

