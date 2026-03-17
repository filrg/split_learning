import os
import pika
import pickle
import sys
import numpy as np
import copy
import src.Log
import src.Utils
import torch

from src.model.Bert_AGNEWS import Bert_AGNEWS
from src.model.KWT_SPEECHCOMMANDS import KWT_SPEECHCOMMANDS
from src.model.VGG16_CIFAR10 import VGG16_CIFAR10
from src.val.get_val import get_val

class Server:
    def __init__(self, config):
        # RabbitMQ
        address = config["rabbit"]["address"]
        username = config["rabbit"]["username"]
        password = config["rabbit"]["password"]
        virtual_host = config["rabbit"]["virtual-host"]

        self.model_name = config["server"]["model"]
        self.data_name = config["server"]["data-name"]
        self.total_clients = config["server"]["clients"]
        self.global_round = config["server"]["global-round"]
        self.round = 1
        self.t_g = config["server"]["t-g"]
        self.t_c = config["server"]["t-c"]
        self.num_cluster = config["server"]["num-cluster"]
        self.cut_layer = config["server"]["cut-layer"]

        # Clients
        self.learning = config["learning"]
        self.data_distribution = config["server"]["data-distribution"]

        # Data distribution
        self.non_iid = self.data_distribution["non-iid"]
        self.num_label = self.data_distribution["num-label"]
        self.num_sample = self.data_distribution["num-sample"]
        self.label_counts = None

        log_path = config["log_path"]

        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue')

        self.current_clients = [0, 0]
        self.register_clients = [0, 0]
        self.check_client = 0
        self.responses = {}  # Save response
        self.list_clients = []
        self.round_result = True
        self.full_state_dict = None

        self.clients_avg = [{} for _ in range(self.num_cluster)]
        self.clients_params = [[] for _ in range(self.num_cluster)]
        self.clients_sizes = [[] for _ in range(self.num_cluster)]

        self.edges_params = [None for _ in range(self.num_cluster)]
        self.edges_sizes = [None for _ in range(self.num_cluster)]

        self.infor_cluster = [[0,0] for _ in range(self.num_cluster)]

        self.channel.basic_qos(prefetch_count=1)
        self.reply_channel = self.connection.channel()
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        debug_mode = config["debug_mode"]
        self.logger = src.Log.Logger(f"{log_path}/app.log", debug_mode)
        self.logger.log_info(f"Application start. Server is waiting for {self.total_clients} clients.")
        src.Log.print_with_color(f"Application start. Server is waiting for {self.total_clients} clients.", "green")

    def distribution(self):
        if self.non_iid:
            # Bert
            # label_distribution = np.array([
            #     [0.25, 0.25, 0.25, 0.25],
            #     [0.25, 0.25, 0.25, 0.25],
            #     [0.25, 0.25, 0.25, 0.25],
            #     [0.25, 0.25, 0.25, 0.25]]
            # )

            # VGG16
            label_distribution = np.array([
                [0.1376, 0.0589, 0.2163, 0.2322, 0.0163, 0.0417, 0.1706, 0.0845, 0.0300, 0.0128],
                [0.0036, 0.0046, 0.1981, 0.0632, 0.1970, 0.1301, 0.1275, 0.0384, 0.2076, 0.0288],
                [0.2172, 0.0002, 0.0328, 0.0597, 0.0733, 0.2204, 0.0000, 0.3430, 0.0489, 0.0050],
                [0.0404, 0.2284, 0.1779, 0.0074, 0.0567, 0.0257, 0.0263, 0.0237, 0.1306, 0.2830],
                [0.0142, 0.0838, 0.0240, 0.1238, 0.0046, 0.3854, 0.0094, 0.1984, 0.0868, 0.0690],
                [0.1417, 0.0446, 0.0137, 0.2550, 0.0164, 0.0559, 0.0004, 0.0666, 0.3834, 0.0247],
                [0.0038, 0.1174, 0.3174, 0.1164, 0.1699, 0.0587, 0.1382, 0.0747, 0.0032, 0.0000],
                [0.1713, 0.1041, 0.0189, 0.0213, 0.0000, 0.1449, 0.2591, 0.1255, 0.1427, 0.0070],
                [0.0004, 0.0171, 0.0764, 0.0000, 0.0762, 0.1191, 0.1538, 0.1362, 0.0002, 0.4206]
            ])

            # SPEEDCOMMANDS
            self.label_counts = (label_distribution * self.num_sample).astype(int)
        else:
            self.label_counts = np.full((self.total_clients[0], self.num_label), self.num_sample // self.num_label)

    def on_request(self, ch, method, props, body):
        message = pickle.loads(body)
        routing_key = props.reply_to
        action = message["action"]
        client_id = message["client_id"]
        layer_id = message["layer_id"]
        self.responses[routing_key] = message

        if action == "REGISTER":
            cluster = message['cluster']
            select = message['select']

            if (str(client_id), layer_id, cluster, select) not in self.list_clients:
                self.list_clients.append((str(client_id), layer_id, cluster, select))

            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")

            self.register_clients[layer_id - 1] += 1

            if self.register_clients == self.total_clients:
                src.Log.print_with_color("All clients are connected. Sending notifications.", "green")

                self.distribution()
                self.cluster_and_selection()
                src.Log.print_with_color(f'List cut point: {self.cut_layer}', 'yellow')
                src.Log.print_with_color(f'Infor clusters: {self.infor_cluster}', 'yellow')

                self.logger.log_info(f"Start training global round {self.round}")
                self.notify_clients()

        elif action == "NOTIFY":
            src.Log.print_with_color("[<<<] Received message from client: {message}", "blue")

            self.check_client += 1

            if self.check_client == self.total_clients[0]:
                self.check_client = 0
                src.Log.print_with_color(f"Received finish training notification all clients.", "yellow")
                client_send = self.round % self.t_c == 0
                edge_send = self.round % self.t_g == 0

                for (client_id, layer_id, _, _) in self.list_clients:
                    if layer_id == 1:
                        message = {"action": "PAUSE", "send": client_send}
                    else:
                        message = {"action": "PAUSE", "send": edge_send}
                    self.send_to_response(client_id, pickle.dumps(message))

        elif action == "UPDATE":
            data_message = message["message"]
            result = message["result"]
            cluster = message["cluster"]
            src.Log.print_with_color(f"[<<<] Received message from {client_id}: {data_message}, cluster: {cluster}", "blue")

            self.current_clients[layer_id - 1] += 1
            if not result:
                self.round_result = False

            if self.round_result:
                model_state_dict = message["parameters"]
                client_size = message["size"]
                if layer_id == 1:
                    self.clients_params[cluster].append(model_state_dict)
                    self.clients_sizes[cluster].append(client_size)
                else:
                    self.edges_params[cluster] = model_state_dict
                    self.edges_sizes[cluster] = client_size

            # If consumed all client's parameters
            if self.current_clients == self.total_clients:
                src.Log.print_with_color("Collected all parameters.", "yellow")
                if self.round_result:
                    if self.round % self.t_c == 0:

                        for idx in range(self.num_cluster):
                            src.Log.print_with_color(f"Avg client params cluster {idx}", "yellow")
                            state_dict = src.Utils.fedavg_state_dicts(self.clients_params[idx], self.clients_sizes[idx])
                            self.clients_avg[idx] = state_dict

                    if self.round % self.t_g == 0:
                        self.concatenate()

                        if not get_val(self.model_name, self.data_name, self.full_state_dict, self.logger):
                            self.logger.log_warning("Validation failed!")
                            self.round = self.global_round + 1
                        else:
                            torch.save(self.full_state_dict, f'{self.model_name}_{self.data_name}.pth')

                    self.round += 1
                    self.current_clients = [0, 0]

                    self.clients_params = [[] for _ in range(self.num_cluster)]
                    self.clients_sizes = [[] for _ in range(self.num_cluster)]

                    self.edges_params = [None for _ in range(self.num_cluster)]
                    self.edges_sizes = [None for _ in range(self.num_cluster)]

                else:
                    self.round = self.global_round + 1

                # Start a new training round
                self.round_result = True

                if self.round <= self. global_round:
                    self.logger.log_info(f"Start training round {self.round}")
                    if self.round % self.t_g == 0:
                        self.notify_clients()
                    elif self.round % self.t_c == 0:

                        self.notify_clients(subround=True,update=True)
                    else:
                        self.notify_clients(subround=True,update=False)
                else:
                    self.logger.log_info("Stop training !!!")
                    self.notify_clients(start=False)
                    sys.exit()

        ch.basic_ack(delivery_tag=method.delivery_tag)


    def notify_clients(self, start=True, subround=False, update=False):
        for (client_id, layer_id, cluster, label) in self.list_clients:
            filepath = f'{self.model_name}_{self.data_name}.pth'
            state_dict = None

            if start:
                if subround:
                    if update:
                        if client_id == 1:
                            state_dict = self.clients_avg[cluster]
                else:
                    if os.path.exists(filepath):
                        self.full_state_dict = torch.load(filepath, weights_only=True)

                        if self.model_name == 'VGG16':
                            klass = VGG16_CIFAR10
                            if layer_id == 1:
                                model = klass(end_layer=self.cut_layer[cluster])
                            else:
                                model = klass(start_layer=self.cut_layer[cluster])
                            state_dict = model.state_dict()
                            keys = state_dict.keys()

                            for key in keys:
                                state_dict[key] = self.full_state_dict[key]
                        elif self.model_name == "KWT":
                            klass = KWT_SPEECHCOMMANDS
                            if layer_id == 1:
                                model = klass(end_layer=self.cut_layer[cluster])
                            else:
                                model = klass(start_layer=self.cut_layer[cluster])
                            state_dict = model.state_dict()
                            keys = state_dict.keys()

                            for key in keys:
                                state_dict[key] = self.full_state_dict[key]
                        else:
                            klass = Bert_AGNEWS
                            if layer_id == 1:
                                model = klass(layer_id=1, n_block=self.cut_layer[cluster])
                                state_dict = model.state_dict()
                                keys = state_dict.keys()

                                for key in keys:
                                    state_dict[key] = self.full_state_dict[key]
                                else:
                                    model = klass(layer_id=2, n_block=12 - self.cut_layer[cluster])
                                    state_dict = model.state_dict()
                                    state_dict = src.Utils.change_keys(state_dict, self.cut_layer[cluster], True)
                                    keys = state_dict.keys()

                                    for key in keys:
                                        state_dict[key] = self.full_state_dict[key]

                                    state_dict = src.Utils.change_keys(state_dict, self.cut_layer[cluster], False)

                                src.Log.print_with_color(f"Load pretrain model Bert successfully", "green")
                    else:
                        self.logger.log_info(f"File {filepath} does not exist.")

                src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")
                response = {"action": "START",
                            "message": "Server accept the connection!",
                            "parameters": state_dict,
                            "cut_layer": self.cut_layer[cluster],
                            "model_name": self.model_name,
                            "data_name": self.data_name,
                            "learning": self.learning,
                            "label_count": label,
                            "cluster": cluster}
                self.send_to_response(client_id, pickle.dumps(response))
            else:
                src.Log.print_with_color(f"[>>>] Sent stop training request to client {client_id}", "red")
                response = {"action": "STOP",
                            "message": "Stop training!",
                            "parameters": None}
                self.send_to_response(client_id, pickle.dumps(response))

    def cluster_and_selection(self):
        self.label_counts = self.label_counts.tolist()
        new_list_client = []
        for idx, (client_id, layer_id, cluster, select) in enumerate(self.list_clients):
            if layer_id == 1:
                if select:
                    new_list_client.append((client_id, layer_id, cluster, self.label_counts.pop(0)))
                    self.infor_cluster[cluster][0] += 1
                else:
                    self.label_counts.pop(0)
                    self.total_clients[0] -= 1
            else:
                new_list_client.append((client_id, layer_id, cluster, []))
                self.infor_cluster[cluster][1] += 1

        self.list_clients = new_list_client

    def start(self):
        self.channel.start_consuming()

    def send_to_response(self, client_id, message):
        reply_queue_name = f'reply_{client_id}'
        self.reply_channel.queue_declare(reply_queue_name, durable=False)

        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
        self.reply_channel.basic_publish(
            exchange='',
            routing_key=reply_queue_name,
            body=message
        )

    def concatenate(self):
        list_state_dict = []
        full_dict = {}
        for idx in range(self.num_cluster):
            full_dict.update(copy.deepcopy(self.clients_avg[idx]))
            if self.model_name == 'Bert':
                sd = src.Utils.change_keys(self.edges_params[idx], self.cut_layer[0], True)
            else:
                sd = self.edges_params[idx]
            full_dict.update(copy.deepcopy(sd))
            list_state_dict.append(full_dict)

        self.full_state_dict = src.Utils.fedavg_state_dicts(list_state_dict)
