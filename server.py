import argparse
import sys
import signal
from src.Server import Server
from src.Utils import delete_old_queues
import src.Log
import yaml

parser = argparse.ArgumentParser(description="Split learning framework with controller.")

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
    server = Server(config)
    server.start()
    src.Log.print_with_color("Ok, ready!", "green")
