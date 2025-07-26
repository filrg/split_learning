import os
import random
import pika
import pickle
import sys
import numpy as np
import copy
import src.Model
import src.Log
import src.Utils
import src.Validation
from src.model import *


class Server:
    def __init__(self, config):
        # RabbitMQ
        address = config["rabbit"]["address"]
        username = config["rabbit"]["username"]
        password = config["rabbit"]["password"]
        virtual_host = config["rabbit"]["virtual-host"]

        self.partition = config["server"]["cluster"]
        self.model_name = config["server"]["model"]
        self.data_name = config["server"]["data-name"]
        self.total_clients = config["server"]["clients"]
        self.list_cut_layers = config["server"]["no-cluster"]["cut-layers"]
        self.global_round = config["server"]["global-round"]
        self.round = self.global_round
        self.validation = config["server"]["validation"]

        # Clients
        self.batch_size = config["learning"]["batch-size"]
        self.lr = config["learning"]["learning-rate"]
        self.momentum = config["learning"]["momentum"]
        self.control_count = config["learning"]["control-count"]
        self.clip_grad_norm = config["learning"]["clip-grad-norm"]
        self.compute_loss = config["learning"]["compute-loss"]
        self.data_distribution = config["server"]["data-distribution"]

        # Data distribution
        self.non_iid = self.data_distribution["non-iid"]
        self.num_label = self.data_distribution["num-label"]
        self.num_sample = self.data_distribution["num-sample"]
        self.refresh_each_round = self.data_distribution["refresh-each-round"]
        self.random_seed = config["server"]["random-seed"]
        self.label_counts = None

        if self.random_seed:
            random.seed(self.random_seed)

        log_path = config["log_path"]

        credentials = pika.PlainCredentials(username, password)
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue')

        self.current_clients = 0
        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.responses = {}  # Save response
        self.list_clients = []
        self.round_result = True

        # Model
        self.global_model_parameters = [[] for _ in range(len(self.total_clients))]
        self.global_client_sizes = [[] for _ in range(len(self.total_clients))]

        # Sequential
        self.edge_device = []
        self.device_begin = []
        self.device_stop = []
        self.avg_state_dict = []
        self.idx = 0


        self.channel.basic_qos(prefetch_count=1)
        self.reply_channel = self.connection.channel()
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        debug_mode = config["debug_mode"]
        self.logger = src.Log.Logger(f"{log_path}/app.log", debug_mode)
        self.logger.log_info(f"Application start. Server is waiting for {self.total_clients} clients.")

    def distribution(self):
        if self.non_iid:
            label_distribution = np.random.dirichlet([self.data_distribution["dirichlet"]["alpha"]] * self.num_label,
                                                     self.total_clients[0])
            # self.label_counts = np.array([[480,480,480,480,480,20,20,20,20,20],
            #                                [480,480,480,480,480,20,20,20,20,20],
            #                                [480,480,480,480,480,20,20,20,20,20],
            #                                [480,480,480,480,480,20,20,20,20,20],
            #                                [20,20,20,20,20,480,480,480,480,480],
            #                                [20,20,20,20,20,480,480,480,480,480],
            #                                [20,20,20,20,20,480,480,480,480,480],
            #                                [20,20,20,20,20,480,480,480,480,480]])
            # self.label_counts = np.array([[400,400,400,400,400,100,100,100,100,100],
            #                                [100,100,100,100,100,400,400,400,400,400]])
            self.label_counts = (label_distribution * self.num_sample).astype(int)
        else:
            self.label_counts = np.full((self.total_clients[0], self.num_label), self.num_sample // self.num_label)

    def on_request(self, ch, method, props, body):
        message = pickle.loads(body)
        routing_key = props.reply_to
        action = message["action"]
        client_id = message["client_id"]
        layer_id = message["layer_id"]
        cluster = 0
        self.responses[routing_key] = message

        if action == "REGISTER":
            performance = message['performance']
            if (str(client_id), layer_id, performance, 0) not in self.list_clients:
                self.list_clients.append((str(client_id), layer_id, performance, 0))
                if layer_id == 1:
                    self.edge_device.append((str(client_id), layer_id, performance, 0))
                else:
                    self.device_begin.append((str(client_id), layer_id, performance, 0))
                    self.device_stop.append((str(client_id), layer_id, performance, 0))
            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            # Save messages from clients
            self.register_clients[layer_id - 1] += 1

            # If consumed all clients - Register for first time
            if self.register_clients == self.total_clients:
                self.device_begin.append(self.edge_device[0])
                self.device_stop.append(self.edge_device[-1])
                self.distribution()
                src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
                self.logger.log_info(f"Start training round {self.global_round - self.round + 1}")
                self.notify_clients()

        elif action == "NOTIFY":
            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            message = {"action": "PAUSE",
                       "message": "Pause training and please send your parameters",
                       "parameters": None}
            if self.idx < self.total_clients[0] - 1:
                (client_id, layer_id, _, clustering) = self.edge_device[self.idx]
                self.send_to_response(client_id, pickle.dumps(message))
            else:
                for (client_id, layer_id, _, clustering) in self.device_stop:
                    self.send_to_response(client_id, pickle.dumps(message))

        elif action == "UPDATE":
            data_message = message["message"]
            result = message["result"]
            model_state_dict = message["parameters"]
            client_size = message["size"]
            src.Log.print_with_color(f"[<<<] Received message from {client_id}: {data_message}", "blue")
            if self.idx < self.total_clients[0] - 1:
                self.global_model_parameters[layer_id - 1].append(model_state_dict)
                self.idx += 1
                self.notify_clients(register=False, idx=self.idx)

            else:
                self.current_clients += 1
                if not result:
                    self.round_result = False

                self.global_model_parameters[layer_id - 1].append(model_state_dict)
                self.global_client_sizes[layer_id - 1].append(client_size)

                if self.current_clients == len(self.device_stop):

                    src.Log.print_with_color("Collected all parameters of final devices.", "yellow")
                    self.current_clients = 0
                    self.idx = 0
                    self.avg_all_parameters()
                    self.global_model_parameters = [[] for _ in range(len(self.total_clients))]
                    self.global_client_sizes = [[] for _ in range(len(self.total_clients))]

                    if self.validation and self.round_result:
                        state_dict_full =  self.concatenate_and_avg_clusters()
                        self.avg_state_dict = []
                        if not src.Validation.test(self.model_name, self.data_name, state_dict_full, self.logger):
                            self.logger.log_warning("Training failed!")
                        else:
                            torch.save(state_dict_full, f'{self.model_name}_{self.data_name}.pth')
                            self.round -= 1
                    else:
                        self.round -= 1

                    self.round_result = True

                    if self.round > 0:
                        self.logger.log_info(f"Start training round {self.global_round - self.round + 1}")
                        self.notify_clients()
                    else:
                        self.notify_clients(start=False)
                        sys.exit()

        ch.basic_ack(delivery_tag=method.delivery_tag)

    def notify_clients(self, start=True, register=True, idx = 0):
        cluster = 0
        if start:
            if register:
                label = self.label_counts[idx]
                if 'MNIST' in self.data_name:
                    klass = globals()[f'{self.model_name}_MNIST']
                else:
                    klass = globals()[f'{self.model_name}_{self.data_name}']
                full_model = klass()
                if self.model_name != 'ViT':
                    full_model = nn.Sequential(*nn.ModuleList(full_model.children()))

                for (client_id, layer_id, _, clustering) in self.device_begin:
                    filepath = f'{self.model_name}_{self.data_name}.pth'
                    state_dict = None

                    if layer_id == 1:
                        layers = [0, self.list_cut_layers[0]]
                    elif layer_id == len(self.total_clients):
                        layers = [self.list_cut_layers[-1], -1]
                    else:
                        layers = [self.list_cut_layers[layer_id - 2],
                                  self.list_cut_layers[layer_id - 1]]

                    if os.path.exists(filepath):
                        full_state_dict = torch.load(filepath, weights_only=True)
                        if self.model_name != 'ViT':
                            full_model.load_state_dict(full_state_dict)

                            if layer_id == 1:
                                if layers == [0, 0]:
                                    model_part = nn.Sequential(*nn.ModuleList(full_model.children())[:])
                                else:
                                    model_part = nn.Sequential(*nn.ModuleList(full_model.children())[:layers[1]])
                            elif layer_id == len(self.total_clients):
                                model_part = nn.Sequential(*nn.ModuleList(full_model.children())[layers[0]:])
                            else:
                                model_part = nn.Sequential(
                                    *nn.ModuleList(full_model.children())[layers[0]:layers[1]])

                            state_dict = model_part.state_dict()
                            self.logger.log_info("Model loaded successfully.")
                        else:
                            if layer_id == 1:
                                if layers == [0, 0]:
                                    model = klass()
                                else:
                                    model = klass(end_layer=layers[1])
                            elif layer_id == len(self.total_clients):
                                model = klass(start_layer=layers[0])
                            else:
                                model = klass(start_layer=layers[0], end_layer=layers[1])
                            state_dict = model.state_dict()
                            keys = state_dict.keys()

                            for key in keys:
                                state_dict[key] = full_state_dict[key]
                            self.logger.log_info("Model loaded successfully.")
                    else:
                        self.logger.log_info(f"File {filepath} does not exist.")

                    src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")

                    response = {"action": "START",
                                "message": "Server accept the connection!",
                                "parameters": copy.deepcopy(state_dict),
                                "num_layers": len(self.total_clients),
                                "layers": layers,
                                "model_name": self.model_name,
                                "data_name": self.data_name,
                                "control_count": self.control_count,
                                "batch_size": self.batch_size,
                                "lr": self.lr,
                                "momentum": self.momentum,
                                "clip_grad_norm": self.clip_grad_norm,
                                "compute_loss": self.compute_loss,
                                "label_count": label,
                                "cluster": clustering,
                                "special": False}

                    self.send_to_response(client_id, pickle.dumps(response))

            else:
                label = self.label_counts[idx]
                layers = [0, self.list_cut_layers[0]]
                (client_id, layer_id, _, clustering) = self.edge_device[idx]
                response = {"action": "START",
                            "message": "Server accept the connection!",
                            "parameters": self.global_model_parameters[layer_id - 1].pop(),
                            "num_layers": len(self.total_clients),
                            "layers": layers,
                            "model_name": self.model_name,
                            "data_name": self.data_name,
                            "control_count": self.control_count,
                            "batch_size": self.batch_size,
                            "lr": self.lr,
                            "momentum": self.momentum,
                            "clip_grad_norm": self.clip_grad_norm,
                            "compute_loss": self.compute_loss,
                            "label_count": label,
                            "cluster": clustering,
                            "special": False}

                self.send_to_response(client_id, pickle.dumps(response))
        else:
            for (client_id, layer_id, _, clustering) in self.list_clients:
                src.Log.print_with_color(f"[>>>] Sent stop training request to client {client_id}", "red")
                response = {"action": "STOP",
                            "message": "Stop training!",
                            "parameters": None}
                self.send_to_response(client_id, pickle.dumps(response))

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

    def avg_all_parameters(self):
        layer_sizes = self.global_client_sizes
        layer_params = self.global_model_parameters

        for layer_idx, list_state_dicts in enumerate(layer_params):
            list_sizes = layer_sizes[layer_idx]
            if not list_state_dicts or not list_sizes:
                self.avg_state_dict.append({})
                continue
            avg_sd = src.Utils.fedavg_state_dicts(list_state_dicts, weights=list_sizes)
            self.avg_state_dict.append(avg_sd)

    def concatenate_and_avg_clusters(self):
        full_dict  = {}
        for idx, layer_dict in enumerate(self.avg_state_dict):
            sd = layer_dict
            if self.model_name != 'ViT' and idx > 0:
                sd = src.Utils.change_state_dict(layer_dict, self.list_cut_layers[idx - 1])
            full_dict.update(copy.deepcopy(sd))

        return full_dict
