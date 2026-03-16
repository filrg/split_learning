import pika
import uuid
import argparse
import yaml

import torch

import src.Log
from src.RpcClient import RpcClient

parser = argparse.ArgumentParser(description="Split learning framework")
parser.add_argument('--layer_id', type=int, required=True, help='ID of layer, start from 1')
parser.add_argument('--device', type=str, required=False, help='Device of client')
parser.add_argument('--c', type=int, required=True, help='ID cluster by device')
parser.add_argument('--s', type=bool, required=True, help='Select/Reject device')

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

if __name__ == "__main__":
    src.Log.print_with_color("[>>>] Client sending registration message to server...", "red")

    data = {"action": "REGISTER", "client_id": client_id, "layer_id": args.layer_id, "cluster": args.c, "select": args.s, "message": "Hello from Client!"}
    client = RpcClient(client_id, args.layer_id, channel, device)
    client.send_to_server(data)
    if args.s:
        client.wait_response()
    else:
        src.Log.print_with_color("Rejected !", "yellow")
