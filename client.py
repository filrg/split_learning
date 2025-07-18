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
parser.add_argument('--event_time', type=bool, default=False, required=False,
                    help='Log event time for debug mode')
parser.add_argument('--performance', type=int, required=False, help='Cluster by device')

parser.add_argument('--attack_mode', choices=['normal', 'pixel', 'semantic'], default='normal',
                    help='Dataset mode: normal, pixel-trigger backdoor, or semantic backdoor')
parser.add_argument('--poison_rate', type=float, default=0.5, help='Fraction of samples to poison')
parser.add_argument('--trigger_size', type=int, default=10, help='Size of pixel trigger square')
parser.add_argument('--trigger_location', choices=['bottom_right', 'bottom_left', 'top_right', 'top_left'],
                    default='bottom_right', help='Location for pixel trigger')
parser.add_argument('--trigger_color', nargs=3, type=float, default=[1.0, 0.0, 0.0],
                    help='RGB color for pixel trigger (list of 3 floats)')
parser.add_argument('--stripe_width', type=int, default=4, help='Width of stripes for semantic backdoor')
parser.add_argument('--alpha', type=float, default=0.5, help='Alpha blending factor for semantic stripes')
parser.add_argument('--stripe_orientation', choices=['vertical', 'horizontal'], default='vertical',
                    help='Orientation of semantic stripes')
parser.add_argument('--target_label', type=int, default=0, help='Target label for poisoned samples')

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

if args.performance is None:
    performance = -1
else:
    performance = args.performance

if __name__ == "__main__":
    src.Log.print_with_color("[>>>] Client sending registration message to server...", "red")
    data = {"action": "REGISTER", "client_id": client_id, "layer_id": args.layer_id, "performance": performance, "message": "Hello from Client!"}
    scheduler = Scheduler(client_id, channel, device, args)
    client = RpcClient(client_id, address, username, password, scheduler.train_on_device, device, args)
    client.send_to_server(data)
    client.wait_response()
