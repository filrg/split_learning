import torch
import os
import random
import pika
import pickle
import sys
import numpy as np
import copy
import src.Log
import src.Utils

from src.model.BERT_AGNEWS import BERT_AGNEWS
from src.model.VGG16_CIFAR10 import VGG16_CIFAR10
from src.model.KWT_SPEECHCOMMANDS import KWT_SPEECHCOMMANDS

from src.Cluster import clustering_algorithm
from src.Selection import auto_threshold
from src.Partition import partition
from src.val.get_val import get_val

class Server:
    def __init__(self, config):
        # RabbitMQ
        address = config["rabbit"]["address"]
        username = config["rabbit"]["username"]
        password = config["rabbit"]["password"]
        virtual_host = config["rabbit"]["virtual-host"]

        self.auto_mode = config["server"]["auto-mode"]
        self.manual = config["server"]["manual"]
        self.cluster_selection = config["server"]["cluster-selection"]
        self.model_name = config["server"]["model"]
        self.data_name = config["server"]["data-name"]
        self.total_clients = config["server"]["clients"]
        self.list_cut_layers = []
        self.global_round = config["server"]["global-round"]
        self.round = self.global_round
        self.save_parameters = config["server"]["parameters"]["save"]
        self.load_parameters = config["server"]["parameters"]["load"]
        self.validation = config["server"]["validation"]

        # Clients
        self.learning = config["learning"]

        # Data distribution
        self.data_distribution = config["server"]["data-distribution"]
        self.refresh = self.data_distribution["refresh"]
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

        self.current_clients = [0 for _ in range(len(self.total_clients))]
        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.first_layer_clients_in_each_cluster = []
        self.responses = {}  # Save response
        self.list_clients = []
        self.round_result = True

        self.global_model_parameters = None
        self.global_client_sizes = None
        self.avg_state_dict = None

        self.size_data = None
        self.num_cluster = 1
        self.infor_cluster = None
        self.reject = False

        self.channel.basic_qos(prefetch_count=10)
        self.reply_channel = self.connection.channel()
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        debug_mode = config["debug_mode"]
        self.logger = src.Log.Logger(f"{log_path}/app.log", debug_mode)
        self.logger.log_info(f"Application start. Server is waiting for {self.total_clients} clients.")

    def distribution(self):
        if self.data_distribution["non-iid"]:
            label_distribution = np.random.dirichlet([self.data_distribution["dirichlet"]["alpha"]] * self.data_distribution['num-label'],
                                                     self.total_clients[0])

            self.label_counts = (label_distribution * self.data_distribution["num-sample"]).astype(int)
        else:
            self.label_counts = np.full((self.total_clients[0], self.data_distribution["num-label"]), self.data_distribution["num-sample"] // self.data_distribution["num-label"])

        label_counts = self.label_counts.tolist()
        for idx, (client_id, layer_id, profile, cluster) in enumerate(self.list_clients):
            if layer_id == 1:
                self.list_clients[idx] = (client_id, layer_id, profile, cluster , label_counts.pop())
            else:
                self.list_clients[idx] = (client_id, layer_id, profile, cluster, [])

    def on_request(self, ch, method, props, body):
        message = pickle.loads(body)
        routing_key = props.reply_to
        action = message["action"]
        client_id = message["client_id"]
        layer_id = message["layer_id"]
        self.responses[routing_key] = message

        if action == "REGISTER":
            profile = message['profile']
            cluster =  message['cluster']

            if self.size_data is None:
                if layer_id == 1:
                    self.size_data = profile['size_data']

            if (str(client_id), layer_id, profile, cluster) not in self.list_clients:
                self.list_clients.append((str(client_id), layer_id, profile, cluster))

            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            self.register_clients[layer_id - 1] += 1

            if self.register_clients == self.total_clients:
                src.Log.print_with_color("All clients are connected. Sending notifications.", "green")

                self.distribution()
                self.cluster_and_selection()
                src.Log.print_with_color(f'List cut point: {self.list_cut_layers}', 'yellow')
                src.Log.print_with_color(f'Infor clusters: {self.infor_cluster}', 'yellow')

                self.logger.log_info(f"Start training round {self.global_round - self.round + 1}")
                self.notify_clients()

        elif action == "NOTIFY":
            cluster = message["cluster"]
            src.Log.print_with_color("[<<<] Received message from client: {message}", "blue")
            message = {"action": "PAUSE",
                       "message": "Pause training and please send your parameters",
                       "parameters": None}
            if layer_id == 1:
                self.first_layer_clients_in_each_cluster[cluster] += 1

            if self.first_layer_clients_in_each_cluster[cluster] == self.infor_cluster[cluster][0]:
                self.first_layer_clients_in_each_cluster[cluster] = 0
                src.Log.print_with_color(f"Received finish training notification cluster {cluster}", "yellow")

                for (client_id, layer_id, _, clustering ,_, train) in self.list_clients:
                    if train is True:
                        if clustering == cluster:
                            self.send_to_response(client_id, pickle.dumps(message))

        elif action == "UPDATE":
            data_message = message["message"]
            result = message["result"]
            src.Log.print_with_color(f"[<<<] Received message from {client_id}: {data_message}", "blue")
            cluster = message["cluster"]

            self.current_clients[layer_id - 1] += 1
            if not result:
                self.round_result = False

            # Save client's model parameters
            if self.save_parameters and self.round_result:
                model_state_dict = message["parameters"]
                client_size = message["size"]
                self.global_model_parameters[cluster][layer_id - 1].append(model_state_dict)
                self.global_client_sizes[cluster][layer_id - 1].append(client_size)

            # If consumed all client's parameters
            if self.current_clients == self.total_clients:
                src.Log.print_with_color("Collected all parameters.", "yellow")
                if self.save_parameters and self.round_result:
                    for i in range(0, self.num_cluster):
                        self.avg_all_parameters(i)
                        self.global_model_parameters[i] = [[] for _ in range(len(self.total_clients))]
                        self.global_client_sizes[i] = [[] for _ in range(len(self.total_clients))]
                self.current_clients = [0 for _ in range(len(self.total_clients))]
                # Test
                if self.save_parameters and self.validation and self.round_result:
                    state_dict_full = self.concatenate_and_avg_clusters()
                    self.avg_state_dict = [[] for _ in range(self.num_cluster)]
                    if not get_val(self.model_name, self.data_name, state_dict_full,self.logger):
                        self.logger.log_warning("Training failed!")
                    else:
                        # Save to files
                        torch.save(state_dict_full, f'{self.model_name}_{self.data_name}.pth')
                        self.round -= 1
                else:
                    self.round -= 1

                # Start a new training round
                self.round_result = True

                if self.round > 0:
                    self.logger.log_info(f"Start training round {self.global_round - self.round + 1}")

                    if self.save_parameters:
                        self.notify_clients()
                    else:
                        self.notify_clients(register=False)
                else:
                    self.logger.log_info("Stop training !!!")
                    self.notify_clients(start=False)
                    sys.exit()

        ch.basic_ack(delivery_tag=method.delivery_tag)

    def notify_clients(self, start=True, register=True, cluster=None):
        if cluster is not None:
            for (client_id, layer_id, _, clustering,_, train) in self.list_clients:
                if train is True:
                    if clustering == cluster:
                        if layer_id == 1:
                            layers = [0, self.list_cut_layers[cluster][0]]
                        elif layer_id == len(self.total_clients):
                            layers = [self.list_cut_layers[cluster][-1], -1]
                        else:
                            layers = [self.list_cut_layers[cluster][layer_id - 2],
                                      self.list_cut_layers[cluster][layer_id - 1]]
                        src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")

                        response = {"action": "START",
                                    "message": "Server accept the connection!",
                                    "parameters": copy.deepcopy(self.avg_state_dict[cluster][layer_id - 1]),
                                    "layers": layers,
                                    "model_name": self.model_name,
                                    "data_name": self.data_name,
                                    "learning": self.learning,
                                    "refresh": self.refresh,
                                    "label_count": None,
                                    "cluster": cluster}

                        self.send_to_response(client_id, pickle.dumps(response))
        else:
            for (client_id, layer_id, _, clustering, label, train) in self.list_clients:

                filepath = f'{self.model_name}_{self.data_name}.pth'
                state_dict = None

                if start:
                    if layer_id == 1:
                        layers = [0, self.list_cut_layers[clustering][0]]
                    elif layer_id == len(self.total_clients):
                        layers = [self.list_cut_layers[clustering][-1], -1]
                    else:
                        layers = [self.list_cut_layers[clustering][layer_id - 2],
                                  self.list_cut_layers[clustering][layer_id - 1]]

                    if self.load_parameters and register:
                        if os.path.exists(filepath):
                            full_state_dict = torch.load(filepath, weights_only=True)

                            if self.model_name == "BERT":
                                klass = BERT_AGNEWS
                            elif self.model_name == "KWT":
                                klass = KWT_SPEECHCOMMANDS
                            else:
                                klass = VGG16_CIFAR10

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

                            src.Log.print_with_color(f"Load model {filepath}.pth successfully", "green")
                        else:
                            src.Log.print_with_color(f"File {filepath} does not exist.", "yellow")

                    src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")
                    if train is True:
                        response = {"action": "START",
                                    "message": "Server accept the connection!",
                                    "parameters": copy.deepcopy(state_dict),
                                    "layers": layers,
                                    "model_name": self.model_name,
                                    "data_name": self.data_name,
                                    "learning": self.learning,
                                    "label_count": label,
                                    "refresh": self.refresh,
                                    "cluster": clustering}
                        self.send_to_response(client_id, pickle.dumps(response))

                    else:
                        if self.reject is False:
                            response = {"action": "STOP",
                                        "message": "Reject Device",
                                        "parameters": None}

                            self.send_to_response(client_id, pickle.dumps(response))

                else:
                    src.Log.print_with_color(f"[>>>] Sent stop training request to client {client_id}", "red")
                    response = {"action": "STOP",
                                "message": "Stop training!",
                                "parameters": None}
                    self.send_to_response(client_id, pickle.dumps(response))

            self.reject = True

    def cluster_and_selection(self):
        # Clustering only device in first layer

        if self.auto_mode:
            self.num_cluster = self.cluster_selection['num-cluster']
            # Clustering of devices of layer
            cluster_labels, infor_cluster = clustering_algorithm(self.label_counts, self.num_cluster)
            cluster_labels = cluster_labels.tolist()

            self.infor_cluster = infor_cluster
            for idx , _ in enumerate(self.infor_cluster):
                self.infor_cluster[idx].append(0)

            list_performance = [[] for _ in range(self.num_cluster)]

            for idx, (client_id, layer_id, profile, cluster, label) in enumerate(self.list_clients):
                if layer_id == 1:
                    new_cluster = cluster_labels.pop()
                    self.list_clients[idx] = (client_id, layer_id, profile, new_cluster, label, True)
                    list_performance[new_cluster].append(profile["speed"])
                else:
                    self.list_clients[idx] = (client_id, layer_id, profile, cluster, label, True)

            # Selection of higher devices in each cluster
            if self.cluster_selection["selection-mode"]:
                thresholds = []
                print(list_performance)
                for performances in list_performance:
                    threshold = auto_threshold(performances)
                    thresholds.append(threshold)
                print(thresholds)

                for idx, (client_id, layer_id, performance, cluster, label, select) in enumerate(self.list_clients):
                    if layer_id == 1:
                        if performance < thresholds[cluster]:
                            self.list_clients[idx] = (client_id, layer_id, performance, cluster, label, False)
                            self.total_clients[0] -= 1
                            self.infor_cluster[cluster][0] -= 1
                            src.Log.print_with_color(f"Remove a device has id: {client_id}", "red")
                    else:
                        self.infor_cluster[cluster][1] += 1
            else:
                for idx, (_, layer_id, _, cluster, _, _) in enumerate(self.list_clients):
                    if layer_id == 2:
                        self.infor_cluster[cluster][1] += 1

            # partition in each cluster

            self.list_cut_layers = []
            for id_cluster in range(self.num_cluster):
                exe_time_layer_1 = []
                net_layer_1 = []
                exe_time_layer_2 = []
                net_layer_2 = []
                for (_, layer_id, profile, cluster,_, _) in self.list_clients:
                    if cluster == id_cluster:
                        if layer_id == 1:
                            exe_time_layer_1.append(profile["exe_time"])
                            net_layer_1.append(profile["network"])
                        else:
                            exe_time_layer_2.append(profile["exe_time"])
                            net_layer_2.append(profile["network"])
                cut_point = partition(exe_time_layer_1, net_layer_1, exe_time_layer_2, net_layer_2, self.size_data)
                self.list_cut_layers.append(cut_point)

        else:
            if self.manual['cluster-mode']:
                for idx, (client_id, layer_id, profile, cluster, label) in enumerate(self.list_clients):
                    self.list_clients[idx] = (client_id, layer_id, profile, cluster, label, True)
                self.num_cluster = self.manual["cluster"]["num-cluster"]
                self.infor_cluster = self.manual["cluster"]["infor-cluster"]
                self.list_cut_layers = self.manual["cluster"]["cut-layers"]
            else:
                for idx, (client_id, layer_id, profile, cluster, label) in enumerate(self.list_clients):
                    self.list_clients[idx] = (client_id, layer_id, profile, 0, label, True)
                self.num_cluster = 1
                self.infor_cluster = [self.total_clients]
                self.list_cut_layers = self.manual["no-cluster"]["cut-layers"]

        self.global_model_parameters = [[[] for _ in range(len(self.total_clients))] for _ in range(self.num_cluster)]
        self.global_client_sizes = [[[] for _ in range(len(self.total_clients))] for _ in range(self.num_cluster)]
        self.avg_state_dict = [[] for _ in range(self.num_cluster)]
        self.first_layer_clients_in_each_cluster = [0 for _ in range(self.num_cluster)]

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

    def avg_all_parameters(self, cluster: int):
        layer_sizes = self.global_client_sizes[cluster]
        layer_params = self.global_model_parameters[cluster]

        for layer_idx, list_state_dicts in enumerate(layer_params):
            list_sizes = layer_sizes[layer_idx]
            if not list_state_dicts or not list_sizes:
                self.avg_state_dict[cluster].append({})
                continue
            avg_sd = src.Utils.fedavg_state_dicts(list_state_dicts, weights=list_sizes)
            self.avg_state_dict[cluster].append(avg_sd)

    def concatenate_and_avg_clusters(self):
        cluster_state_dicts = []

        for c in range(self.num_cluster):
            avg_layers = self.avg_state_dict[c] or []
            if not avg_layers:
                print(f"Warning: cluster {c} has no averaged layers, skipping.")
                continue

            full_dict = {}
            if self.list_cut_layers[c][0] != 0:
                for idx, layer_dict in enumerate(avg_layers):
                    sd = layer_dict
                    full_dict.update(copy.deepcopy(sd))
            else:
                full_dict.update(copy.deepcopy(avg_layers[0]))

            cluster_state_dicts.append(full_dict)

        if not cluster_state_dicts:
            raise RuntimeError("There is no cluster to merge and average.")

        global_state = src.Utils.fedavg_state_dicts(cluster_state_dicts)

        return global_state
