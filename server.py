import argparse
import sys
import signal
from src.Server import Server
from src.Utils import delete_old_queues
import src.Log
import yaml

parser = argparse.ArgumentParser(description="Split learning framework with controller.")

parser.add_argument('--attack_mode', choices=['normal', 'pixel', 'semantic'], default='normal',
                    help='Dataset mode: normal, pixel-trigger backdoor, or semantic backdoor')
parser.add_argument('--attack_round', type=int, default=0, required=False,
                    help='Starting round for client attack')
# Pixel trigger args
parser.add_argument('--trigger_size', type=int, default=10, help='Size of pixel trigger square')
parser.add_argument('--trigger_location', choices=['bottom_right', 'bottom_left', 'top_right', 'top_left'],
                    default='bottom_right', help='Location for pixel trigger')
parser.add_argument('--trigger_color', nargs=3, type=float, default=[1.0, 0.0, 0.0],
                    help='RGB color for pixel trigger (list of 3 floats)')
parser.add_argument('--trigger_value', type=float, default=1.0,
                    help='Grayscale value for MNIST trigger')
# Semantic trigger args
parser.add_argument('--stripe_width', type=int, default=4, help='Width of stripes for semantic backdoor')
parser.add_argument('--alpha', type=float, default=0.5, help='Alpha blending factor for semantic stripes')
parser.add_argument('--stripe_orientation', choices=['vertical', 'horizontal'], default='vertical',
                    help='Orientation of semantic stripes')
# Label mapping
parser.add_argument('--label_mapping', type=str, default='',
                    help='Mapping for orig->target labels, e.g. "0:5,1:3". Only backdoored samples remapped.')

args = parser.parse_args()

with open('config.yaml') as file:
    config = yaml.safe_load(file)
address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]
virtual_host = config["rabbit"]["virtual-host"]


def signal_handler(sig, frame):
    print("\nCatch stop signal Ctrl+C. Stop the program.")
    delete_old_queues(address, username, password, virtual_host)
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    delete_old_queues(address, username, password, virtual_host)
    server = Server(config, args)
    server.start()
    src.Log.print_with_color("Ok, ready!", "green")
