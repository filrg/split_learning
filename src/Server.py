import os
import random
import pika
import pickle
import sys
import numpy as np
import torch
import torch.nn as nn
import copy
import src.Model
import src.Log
import src.Utils
from src.Cluster import clustering_algorithm
import src.Validation


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
        self.list_cut_layers = [config["server"]["no-cluster"]["cut-layers"]]
        self.local_round = config["server"]["local-round"]
        self.global_round = config["server"]["global-round"]
        self.round = self.global_round
        self.save_parameters = config["server"]["parameters"]["save"]
        self.load_parameters = config["server"]["parameters"]["load"]
        self.validation = config["server"]["validation"]

        # Clients
        self.batch_size = config["learning"]["batch-size"]
        self.lr = config["learning"]["learning-rate"]
        self.momentum = config["learning"]["momentum"]
        self.control_count = config["learning"]["control-count"]
        self.clip_grad_norm = config["learning"]["clip-grad-norm"]
        self.compute_loss = config["learning"]["compute-loss"]
        self.data_distribution = config["server"]["data-distribution"]

        # Cluster
        self.client_cluster_config = config["server"]["client-cluster"]
        self.mode_cluster = self.client_cluster_config["enable"]
        self.special = self.client_cluster_config["special"]
        self.mode_partition = self.client_cluster_config["auto-partition"]
        if not self.mode_cluster:
            self.local_round = 1

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
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue='rpc_queue')

        self.current_clients = [0 for _ in range(len(self.total_clients))]
        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.first_layer_clients_in_each_cluster = []
        self.responses = {}  # Save response
        self.list_clients = []
        self.global_avg_state_dict = [[] for _ in range(len(self.total_clients))]
        self.round_result = True

        self.global_model_parameters = [[] for _ in range(len(self.total_clients))]
        self.global_client_sizes = [[] for _ in range(len(self.total_clients))]
        self.local_model_parameters = None
        self.local_client_sizes = None
        self.local_avg_state_dict = None
        self.total_cluster_size = None

        self.num_cluster = None
        self.current_local_training_round = None
        self.infor_cluster = None
        self.current_infor_cluster = None
        self.local_update_count = 0

        self.channel.basic_qos(prefetch_count=1)
        self.reply_channel = self.connection.channel()
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        debug_mode = config["debug_mode"]
        self.logger = src.Log.Logger(f"{log_path}/app.log", debug_mode)
        self.logger.log_info(f"Application start. Server is waiting for {self.total_clients} clients.")

    def distribution(self):
        if self.non_iid:
            label_distribution = np.random.dirichlet([self.data_distribution["dirichlet"]["alpha"]] * self.num_label, self.total_clients[0])
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
            performance = message['performance']
            if (str(client_id), layer_id, performance, 0) not in self.list_clients:
                self.list_clients.append((str(client_id), layer_id, performance, -1))

            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            # Save messages from clients
            self.register_clients[layer_id - 1] += 1

            # If consumed all clients - Register for first time
            if self.register_clients == self.total_clients:
                self.distribution()
                self.cluster_client()
                print(self.list_cut_layers)
                src.Log.print_with_color("All clients are connected. Sending notifications.", "green")
                self.logger.log_info(f"Start training round {self.global_round - self.round + 1}")
                self.notify_clients()
        elif action == "NOTIFY":
            cluster = message["cluster"]
            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            message = {"action": "PAUSE",
                       "message": "Pause training and please send your parameters",
                       "parameters": None}
            if layer_id == 1:
                self.first_layer_clients_in_each_cluster[cluster] += 1

            if self.first_layer_clients_in_each_cluster[cluster] == self.infor_cluster[cluster][0]:
                self.first_layer_clients_in_each_cluster[cluster] = 0
                src.Log.print_with_color(f"Received finish training notification cluster {cluster}", "yellow")

                for (client_id, layer_id, _, clustering) in self.list_clients:
                    if clustering == cluster:
                        if self.special is False:
                            self.send_to_response(client_id, pickle.dumps(message))
                        else:
                            if layer_id == 1:
                                self.send_to_response(client_id, pickle.dumps(message))
                self.local_update_count += 1

            if self.special and self.local_update_count == self.num_cluster * self.local_round:
                self.local_update_count = 0
                for (client_id, layer_id, _) in self.list_clients:
                    if layer_id != 1:
                        self.send_to_response(client_id, pickle.dumps(message))

        elif action == "UPDATE":
            # self.distribution()
            data_message = message["message"]
            result = message["result"]
            src.Log.print_with_color(f"[<<<] Received message from {client_id}: {data_message}", "blue")
            cluster = message["cluster"]
            # Global update
            if self.current_local_training_round[cluster] == self.local_round - 1:
                self.current_clients[layer_id - 1] += 1
                if not result:
                    self.round_result = False

                # Save client's model parameters
                if self.save_parameters and self.round_result:
                    model_state_dict = message["parameters"]
                    client_size = message["size"]
                    self.local_model_parameters[cluster][layer_id - 1].append(model_state_dict)
                    self.local_client_sizes[cluster][layer_id - 1].append(client_size)

                # If consumed all client's parameters
                if self.current_clients == self.total_clients:
                    src.Log.print_with_color("Collected all parameters.", "yellow")
                    if self.save_parameters and self.round_result:
                        for i in range(0, self.num_cluster):
                            self.total_cluster_size[i] = sum(self.local_client_sizes[i][0])
                            self.avg_all_parameters(i)
                            self.local_model_parameters[i] = [[] for _ in range(len(self.total_clients))]
                            self.local_client_sizes[i] = [[] for _ in range(len(self.total_clients))]
                    self.current_clients = [0 for _ in range(len(self.total_clients))]
                    self.current_local_training_round = [0 for _ in range(self.num_cluster)]
                    # Test
                    if self.save_parameters and self.validation and self.round_result:
                        state_dict_full = self.concatenate_state_dict()
                        if not src.Validation.test(self.model_name, self.data_name, state_dict_full, self.logger):
                            self.logger.log_warning("Training failed!")
                        else:
                            # Save to files
                            torch.save(state_dict_full, f'{self.model_name}.pth')
                            self.round -= 1
                    else:
                        self.round -= 1

                    # Start a new training round
                    self.round_result = True

                    if self.round > 0:
                        self.logger.log_info(f"Start training round {self.global_round - self.round + 1}")
                        if self.save_parameters:
                            self.notify_clients(special=self.special)
                        else:
                            self.notify_clients(register=False, special=self.special)
                    else:
                        self.logger.log_info("Stop training !!!")
                        self.notify_clients(start=False)
                        sys.exit()

            # Local update
            else:
                if not result:
                    self.round_result = False
                if self.round_result:
                    model_state_dict = message["parameters"]
                    client_size = message["size"]
                    self.local_model_parameters[cluster][layer_id - 1].append(model_state_dict)
                    self.local_client_sizes[cluster][layer_id - 1].append(client_size)
                self.current_infor_cluster[cluster][layer_id - 1] += 1

                if self.special is False:
                    if self.current_infor_cluster[cluster] == self.infor_cluster[cluster]:
                        self.avg_all_parameters(cluster=cluster)
                        self.notify_clients(cluster=cluster, special=False)
                        self.current_local_training_round[cluster] += 1

                        self.local_model_parameters[cluster] = [[] for _ in range(len(self.total_clients))]
                        self.local_client_sizes[cluster] = [[] for _ in range(len(self.total_clients))]
                        self.current_infor_cluster[cluster] = [0 for _ in range(len(self.total_clients))]
                else:
                    if self.current_infor_cluster[cluster][0] == self.infor_cluster[cluster][0]:
                        self.avg_all_parameters(cluster=cluster)
                        self.notify_clients(cluster=cluster, special=True)
                        self.current_local_training_round[cluster] += 1

                        self.local_model_parameters[cluster] = [[] for _ in range(len(self.total_clients))]
                        self.local_client_sizes[cluster] = [[] for _ in range(len(self.total_clients))]
                        self.current_infor_cluster[cluster] = [0 for _ in range(len(self.total_clients))]

        ch.basic_ack(delivery_tag=method.delivery_tag)

    def notify_clients(self, start=True, register=True, cluster=None, special=False):
        label_counts = copy.copy(self.label_counts)
        label_counts = label_counts.tolist()
        if cluster is not None and special is False:
            for (client_id, layer_id, _, clustering) in self.list_clients:
                if clustering == cluster:
                    if layer_id == 1:
                        layers = [0, self.list_cut_layers[cluster][0]]
                    elif layer_id == len(self.total_clients):
                        layers = [self.list_cut_layers[cluster][-1], -1]
                    else:
                        layers = [self.list_cut_layers[cluster][layer_id - 2], self.list_cut_layers[cluster][layer_id - 1]]
                    src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")
                    if layer_id == 1:
                        response = {"action": "START",
                                    "message": "Server accept the connection!",
                                    "parameters": self.local_avg_state_dict[cluster][layer_id - 1],
                                    "num_layers": len(self.total_clients),
                                    "layers": layers,
                                    "model_name": self.model_name,
                                    "data_name": self.data_name,
                                    "control_count": self.control_count,
                                    "batch_size": self.batch_size,
                                    "lr": self.lr,
                                    "momentum": self.momentum,
                                    "compute_loss": self.compute_loss,
                                    "clip_grad_norm": self.clip_grad_norm,
                                    "label_count": None,
                                    "cluster": None,
                                    "special": False}
                    else:
                        response = {"action": "START",
                                    "message": "Server accept the connection!",
                                    "parameters": self.local_avg_state_dict[cluster][layer_id - 1],
                                    "num_layers": len(self.total_clients),
                                    "layers": layers,
                                    "model_name": self.model_name,
                                    "data_name": None,
                                    "control_count": self.control_count,
                                    "batch_size": self.batch_size,
                                    "lr": self.lr,
                                    "momentum": self.momentum,
                                    "compute_loss": self.compute_loss,
                                    "clip_grad_norm": self.clip_grad_norm,
                                    "label_count": None,
                                    "cluster": None,
                                    "special": False}
                    self.send_to_response(client_id, pickle.dumps(response))
        if cluster is None:
            # Send message to clients when consumed all clients
            klass = getattr(src.Model, self.model_name)
            full_model = klass()
            full_model = nn.Sequential(*nn.ModuleList(full_model.children()))
            for (client_id, layer_id, _, clustering) in self.list_clients:
                # Read parameters file
                filepath = f'{self.model_name}.pth'
                state_dict = None

                if start:
                    if layer_id == 1:
                        layers = [0, self.list_cut_layers[clustering][0]]
                    elif layer_id == len(self.total_clients):
                        layers = [self.list_cut_layers[clustering][-1], -1]
                    else:
                        layers = [self.list_cut_layers[clustering][layer_id - 2], self.list_cut_layers[clustering][layer_id - 1]]

                    if self.load_parameters and register:
                        if os.path.exists(filepath):
                            full_state_dict = torch.load(filepath, weights_only=True)
                            full_model.load_state_dict(full_state_dict)

                            if layer_id == 1:
                                if layers == [0, 0]:
                                    model_part = nn.Sequential(*nn.ModuleList(full_model.children())[:])
                                else:
                                    model_part = nn.Sequential(*nn.ModuleList(full_model.children())[:layers[1]])
                            elif layer_id == len(self.total_clients):
                                model_part = nn.Sequential(*nn.ModuleList(full_model.children())[layers[0]:])
                            else:
                                model_part = nn.Sequential(*nn.ModuleList(full_model.children())[layers[0]:layers[1]])

                            state_dict = model_part.state_dict()
                            self.logger.log_info("Model loaded successfully.")
                        else:
                            self.logger.log_info(f"File {filepath} does not exist.")

                    src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")
                    if layer_id == 1:
                        response = {"action": "START",
                                    "message": "Server accept the connection!",
                                    "parameters": state_dict,
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
                                    "label_count": label_counts.pop(),
                                    "cluster": clustering,
                                    "special": self.special}
                    else:
                        response = {"action": "START",
                                    "message": "Server accept the connection!",
                                    "parameters": state_dict,
                                    "num_layers": len(self.total_clients),
                                    "layers": layers,
                                    "model_name": self.model_name,
                                    "data_name": None,
                                    "control_count": self.control_count,
                                    "batch_size": self.batch_size,
                                    "lr": self.lr,
                                    "momentum": self.momentum,
                                    "clip_grad_norm": self.clip_grad_norm,
                                    "compute_loss": self.compute_loss,
                                    "label_count": None,
                                    "cluster": clustering,
                                    "special": self.special}

                else:
                    src.Log.print_with_color(f"[>>>] Sent stop training request to client {client_id}", "red")
                    response = {"action": "STOP",
                                "message": "Stop training!",
                                "parameters": None}
                self.send_to_response(client_id, pickle.dumps(response))
        if cluster is not None and special is True:
            for (client_id, layer_id, _, clustering) in self.list_clients:
                if clustering == cluster:
                    if layer_id == 1:
                        layers = [0, self.list_cut_layers[cluster][0]]
                    elif layer_id == len(self.total_clients):
                        layers = [self.list_cut_layers[cluster][-1], -1]
                    else:
                        layers = [self.list_cut_layers[cluster][layer_id - 2], self.list_cut_layers[cluster][layer_id - 1]]

                    src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")
                    if layer_id == 1:
                        response = {"action": "START",
                                    "message": "Server accept the connection!",
                                    "parameters": self.local_avg_state_dict[cluster][layer_id - 1],
                                    "num_layers": len(self.total_clients),
                                    "layers": layers,
                                    "model_name": self.model_name,
                                    "data_name": self.data_name,
                                    "control_count": self.control_count,
                                    "batch_size": self.batch_size,
                                    "lr": self.lr,
                                    "momentum": self.momentum,
                                    "compute_loss": self.compute_loss,
                                    "clip_grad_norm": self.clip_grad_norm,
                                    "label_count": None,
                                    "cluster": None,
                                    "special": True}
                        self.send_to_response(client_id, pickle.dumps(response))

    def cluster_client(self):
        list_performance = [-1 for _ in range(len(self.list_clients))]
        for idx, (client_id, layer_id, performance, cluster) in enumerate(self.list_clients):
            list_performance[idx] = performance
        # Phân cụm ở đây chỉ layer đầu
        if self.mode_cluster is True:
            self.logger.log_debug(f"mode_partition is {self.mode_partition}")
            if self.mode_partition is True:
                list_cluster, infor_cluster, num_cluster, list_cut_layers = clustering_algorithm(list_performance, self.total_clients[1], self.client_cluster_config, None)
            else:
                list_cluster, infor_cluster, num_cluster, list_cut_layers = clustering_algorithm(list_performance, self.total_clients[1], self.client_cluster_config, self.partition)

            self.infor_cluster = infor_cluster
            self.num_cluster = num_cluster
            self.list_cut_layers = list_cut_layers
        else:
            list_cluster = [0 for _ in range(len(list_performance))]
            self.num_cluster = 1
            self.infor_cluster = [self.total_clients]
        for idx, (client_id, layer_id, performance, cluster) in enumerate(self.list_clients):
            self.list_clients[idx] = (client_id, layer_id, performance, list_cluster[idx])

        self.local_model_parameters = [[[] for _ in range(len(self.total_clients))] for _ in range(self.num_cluster)]
        self.local_client_sizes = [[[] for _ in range(len(self.total_clients))] for _ in range(self.num_cluster)]
        self.local_avg_state_dict = [[[] for _ in range(len(self.total_clients))] for _ in range(self.num_cluster)]
        self.total_cluster_size = [0 for _ in range(self.num_cluster)]
        if self.mode_cluster:
            self.first_layer_clients_in_each_cluster = [0 for _ in range(self.num_cluster)]
        else:
            self.first_layer_clients_in_each_cluster = [0]
        self.current_infor_cluster = [[0] * len(row) for row in self.infor_cluster]
        self.current_local_training_round = [0 for _ in range(len(self.infor_cluster))]

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

    def avg_all_parameters(self, cluster=None):
        size = self.local_client_sizes[cluster]
        parameters = self.local_model_parameters[cluster]
        for layer, state_dicts in enumerate(parameters):
            local_layer_client_size = size[layer]
            num_models = len(state_dicts)
            if num_models == 0:
                return

            denominator = sum(local_layer_client_size)
            if denominator == 0:
                print(f"Warning: denominator is zero at layer {layer}, skipping...")
                continue

            self.local_avg_state_dict[cluster][layer] = state_dicts[0]

            for key in state_dicts[0].keys():
                for i in range(num_models):
                    if torch.isnan(state_dicts[i][key]).any():
                        print(f"Warning: NaN detected in {key} at model {i}, replacing with zero.")
                        state_dicts[i][key] = torch.nan_to_num(state_dicts[i][key])

                if state_dicts[0][key].dtype != torch.long:
                    self.local_avg_state_dict[cluster][layer][key] = sum(
                        state_dicts[i][key].float() * local_layer_client_size[i]
                        for i in range(num_models)
                    ) / denominator
                else:
                    self.local_avg_state_dict[cluster][layer][key] = sum(
                        state_dicts[i][key] * local_layer_client_size[i]
                        for i in range(num_models)
                    ) // denominator

    def concatenate_state_dict(self):
        state_dict_cluster = {}
        list_state_dict_cluster = [state_dict_cluster for _ in range(self.num_cluster)]
        for cluster in range(self.num_cluster):
            if self.list_cut_layers[cluster][0] != 0:
                for i, state_dicts in enumerate(self.local_avg_state_dict[cluster]):
                    if i > 0:
                        state_dicts = src.Utils.change_state_dict(state_dicts, self.list_cut_layers[cluster][i - 1])
                    list_state_dict_cluster[cluster].update(state_dicts)
            else:
                list_state_dict_cluster[cluster].update(self.local_avg_state_dict[cluster][0])

        # Avg all cluster
        state_dict_full = list_state_dict_cluster[0]
        for key in list_state_dict_cluster[0].keys():
            if list_state_dict_cluster[0][key].dtype != torch.long:
                state_dict_full[key] = sum(
                    list_state_dict_cluster[i][key]
                    for i in range(self.num_cluster)) / self.num_cluster
            else:
                state_dict_full[key] = sum(
                    list_state_dict_cluster[i][key]
                    for i in range(self.num_cluster)) // self.num_cluster

        return state_dict_full
