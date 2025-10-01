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

from src.Cluster import clustering_algorithm
from src.Selection import auto_threshold
from src.Selection import lpt
from algorithm.partition import partition
from src.model import *


class Server:
    def __init__(self, config):
        # RabbitMQ
        address = config["rabbit"]["address"]
        username = config["rabbit"]["username"]
        password = config["rabbit"]["password"]
        virtual_host = config["rabbit"]["virtual-host"]

        self.manual_cluster = config["server"]["manual-cluster"]
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
        self.auto_cluster = self.client_cluster_config["auto-cluster"]
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
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials))
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

        self.size_data = None
        self.num_cluster = None
        self.current_local_training_round = None
        self.infor_cluster = None
        self.current_infor_cluster = None
        self.local_update_count = 0
        self.reject = False

        self.channel.basic_qos(prefetch_count=1)
        self.reply_channel = self.connection.channel()
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        debug_mode = config["debug_mode"]
        self.logger = src.Log.Logger(f"{log_path}/app.log", debug_mode)
        # self.logger.log_info(f"Application start. Server is waiting for {self.total_clients} clients.")
        src.Log.print_with_color(f"Application start. Server is waiting for {self.total_clients} clients.", "green")

    def distribution(self):
        if self.non_iid:
            label_distribution = np.random.dirichlet([self.data_distribution["dirichlet"]["alpha"]] * self.num_label,
                                                     self.total_clients[0])
            # label_distribution = np.array([
            #     [1.88670492e-04, 2.78323802e-01, 1.13726152e-02, 1.24349201e-01,
            #      5.38795539e-04, 1.90184440e-01, 6.43625090e-02, 2.54724837e-01,
            #      7.59550521e-02, 7.76595921e-08],
            #
            #     [7.86644625e-06, 1.43142597e-09, 9.23733736e-02, 1.41117583e-03,
            #      5.57904425e-01, 4.59899388e-03, 2.67154361e-02, 6.79897381e-02,
            #      1.90655112e-04, 2.48808334e-01],
            #
            #     [1.28974551e-03, 7.09386721e-02, 1.48008106e-01, 1.34067660e-01,
            #      6.63072091e-02, 2.32514706e-04, 9.58605190e-04, 3.68931950e-03,
            #      4.65252553e-02, 5.27982912e-01],
            #
            #     [1.47841723e-02, 8.78359666e-03, 4.55327204e-01, 1.57679071e-06,
            #      2.85479330e-01, 9.25068105e-04, 2.12506510e-01, 1.36979009e-03,
            #      1.40011779e-02, 6.82157427e-03],
            #
            #     [1.90630425e-06, 2.25567241e-02, 8.21096538e-01, 3.25336373e-05,
            #      1.31714757e-03, 5.89168566e-02, 1.69602744e-06, 6.43260198e-06,
            #      8.83353304e-09, 9.60701568e-02],
            #
            #     [3.53581329e-01, 8.26254786e-02, 1.01462752e-03, 3.96087356e-01,
            #      1.60483180e-04, 4.51104455e-03, 4.13400181e-02, 9.78326450e-07,
            #      7.11202637e-06, 1.20671573e-01],
            #
            #     [9.49379988e-05, 4.65601685e-02, 6.32840557e-03, 4.47674790e-03,
            #      8.49202122e-02, 8.56501663e-09, 2.58490916e-01, 3.96866660e-01,
            #      2.01766516e-01, 4.95426887e-04],
            #
            #     [2.08492514e-02, 4.09489645e-02, 9.72125652e-02, 1.90336896e-02,
            #      3.90847431e-04, 1.29940518e-01, 4.96233948e-04, 6.49792721e-01,
            #      3.98916100e-02, 1.44359867e-03],
            #
            #     [7.07523146e-02, 4.22245806e-08, 4.25533228e-02, 3.42306651e-04,
            #      7.68246482e-01, 7.34896463e-02, 4.41445349e-02, 4.66822655e-04,
            #      6.75889808e-12, 4.52780196e-06],
            #
            #     [9.30039177e-05, 9.77639318e-06, 2.95348498e-02, 2.12082532e-02,
            #      1.18523841e-14, 8.28632986e-04, 8.09996300e-01, 1.24028178e-03,
            #      1.36135136e-01, 9.53766212e-04]
            # ])

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
            cluster =  message['cluster']
            performance = message['performance']
            exe_time = message['exe_time']
            net = message['net']
            size_data = message['size_data']
            if self.size_data is None:
                self.size_data = size_data

            if (str(client_id), layer_id, performance, cluster, exe_time, net) not in self.list_clients:
                self.list_clients.append((str(client_id), layer_id, performance, cluster, exe_time, net))

            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            # Save messages from clients
            self.register_clients[layer_id - 1] += 1

            # If consumed all clients - Register for first time
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
            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            message = {"action": "PAUSE",
                       "message": "Pause training and please send your parameters",
                       "parameters": None}
            if layer_id == 1:
                self.first_layer_clients_in_each_cluster[cluster] += 1

            if self.first_layer_clients_in_each_cluster[cluster] == self.infor_cluster[cluster][0]:
                self.first_layer_clients_in_each_cluster[cluster] = 0
                src.Log.print_with_color(f"Received finish training notification cluster {cluster}", "yellow")

                for (client_id, layer_id, _, clustering, _, _ ,_, train) in self.list_clients:
                    if train is True:
                        if clustering == cluster:
                            if self.special is False:
                                self.send_to_response(client_id, pickle.dumps(message))
                            else:
                                if layer_id == 1:
                                    self.send_to_response(client_id, pickle.dumps(message))
                self.local_update_count += 1

            if self.special and self.local_update_count == self.num_cluster * self.local_round:
                self.local_update_count = 0
                for (client_id, layer_id, _, clustering, _, _ ,_, train) in self.list_clients:
                    if train is True:
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
                        state_dict_full = self.concatenate_and_avg_clusters()
                        self.local_avg_state_dict = [[] for _ in range(self.num_cluster)]
                        if not src.Validation.test(self.model_name, self.data_name, state_dict_full, self.logger):
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
                        self.local_avg_state_dict[cluster] = []
                        self.current_local_training_round[cluster] += 1

                        self.local_model_parameters[cluster] = [[] for _ in range(len(self.total_clients))]
                        self.local_client_sizes[cluster] = [[] for _ in range(len(self.total_clients))]
                        self.current_infor_cluster[cluster] = [0 for _ in range(len(self.total_clients))]
                else:
                    if self.current_infor_cluster[cluster][0] == self.infor_cluster[cluster][0]:
                        self.avg_all_parameters(cluster=cluster)
                        self.notify_clients(cluster=cluster, special=True)
                        self.local_avg_state_dict[cluster] = []
                        self.current_local_training_round[cluster] += 1

                        self.local_model_parameters[cluster] = [[] for _ in range(len(self.total_clients))]
                        self.local_client_sizes[cluster] = [[] for _ in range(len(self.total_clients))]
                        self.current_infor_cluster[cluster] = [0 for _ in range(len(self.total_clients))]

        ch.basic_ack(delivery_tag=method.delivery_tag)

    def notify_clients(self, start=True, register=True, cluster=None, special=False):

        if cluster is not None and special is False:
            for (client_id, layer_id, _, clustering, _, _ ,_, train) in self.list_clients:
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
                                    "parameters": copy.deepcopy(self.local_avg_state_dict[cluster][layer_id - 1]),
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

                        self.send_to_response(client_id, pickle.dumps(response))

        if cluster is None:

            # Send message to clients when consumed all clients
            if 'MNIST' in self.data_name:
                klass = globals()[f'{self.model_name}_MNIST']
            else:
                klass = globals()[f'{self.model_name}_{self.data_name}']
            full_model = klass()
            if self.model_name != 'ViT':
                full_model = nn.Sequential(*nn.ModuleList(full_model.children()))

            for (client_id, layer_id, _, clustering, _, _, label, train) in self.list_clients:
                # Read parameters file
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

                        else:
                            self.logger.log_info(f"File {filepath} does not exist.")

                    src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")
                    if train is True:
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
                                    "special": self.special}
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

        if cluster is not None and special is True:
            for (client_id, layer_id, _, clustering) in self.list_clients:
                if clustering == cluster:
                    if layer_id == 1:
                        layers = [0, self.list_cut_layers[cluster][0]]
                    elif layer_id == len(self.total_clients):
                        layers = [self.list_cut_layers[cluster][-1], -1]
                    else:
                        layers = [self.list_cut_layers[cluster][layer_id - 2],
                                  self.list_cut_layers[cluster][layer_id - 1]]

                    src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")
                    if layer_id == 1:
                        response = {"action": "START",
                                    "message": "Server accept the connection!",
                                    "parameters": copy.deepcopy(self.local_avg_state_dict[cluster][layer_id - 1]),
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

    def cluster_and_selection(self):
        # Clustering only device in first layer

        if self.mode_cluster is True:
            self.logger.log_debug(f"Auto_partition is {self.auto_cluster}")
            if self.auto_cluster is True:

                # Clustering of devices of layer
                cluster_labels, infor_cluster, num_cluster = clustering_algorithm(self.label_counts,self.client_cluster_config)

                cluster_labels = cluster_labels.tolist()
                self.label_counts = self.label_counts.tolist()
                self.infor_cluster = infor_cluster
                self.num_cluster = num_cluster
                list_performance = [[] for _ in range(num_cluster)]

                for idx, (client_id, layer_id, performance, cluster, exe_time, net) in enumerate(self.list_clients):
                    if layer_id == 1:
                        new_cluster = cluster_labels.pop()
                        self.list_clients[idx] = (client_id, layer_id, performance, new_cluster, exe_time, net, self.label_counts.pop(), True)
                        list_performance[new_cluster].append(performance)

                    else:
                        self.list_clients[idx] = (client_id, layer_id, performance, -1, exe_time, net, [], True)

                # Selection of higher devices in each cluster
                thresholds = []
                print(list_performance)
                for performances in list_performance:
                    threshold = auto_threshold(performances)
                    thresholds.append(threshold)
                print(thresholds)
                total_performance_cluster = [0 for _ in range(self.num_cluster)]
                performance_device_layer_2 = []

                for idx, (client_id, layer_id, performance, cluster, exe_time, net, label, select) in enumerate(self.list_clients):
                    if layer_id == 1:
                        if performance < thresholds[cluster]:
                            self.list_clients[idx] = (client_id, layer_id, performance, cluster, exe_time, net ,label, False)
                            self.total_clients[0] -= 1
                            self.infor_cluster[cluster][0] -= 1
                            src.Log.print_with_color(f"Remove a device has id: {client_id}", "red")

                        else:
                            total_performance_cluster[cluster] += performance

                    else:
                        performance_device_layer_2.append((idx, performance))

                # selecting of devices in layer 2
                infor_layer_2 = lpt(performance_device_layer_2, total_performance_cluster, self.num_cluster)

                for (cluster_layer_2, layer2) in infor_layer_2:
                    count = 0
                    for device in layer2:
                        (client_id, layer_id, performance, cluster, exe_time, net ,label, train) = self.list_clients[device]
                        self.list_clients[device] =  (client_id, layer_id, performance, cluster_layer_2, exe_time, net ,label, True)
                        count += 1
                    self.infor_cluster[cluster_layer_2].append(count)

                # partition in each cluster

                self.list_cut_layers = []
                for id_cluster in range(self.num_cluster):
                    exe_time_layer_1 = []
                    net_layer_1 = []
                    exe_time_layer_2 = []
                    net_layer_2 = []
                    for (client_id, layer_id, performance, cluster, exe_time, net ,label, train) in self.list_clients:
                        if cluster == id_cluster:
                            if layer_id == 1:
                                exe_time_layer_1.append(exe_time)
                                net_layer_1.append(net)
                            else:
                                exe_time_layer_2.append(exe_time)
                                net_layer_2.append(net)
                    cut_point = partition(exe_time_layer_1, net_layer_1, exe_time_layer_2, net_layer_2, self.size_data)
                    self.list_cut_layers.append(cut_point)

            else:
                # self.label_counts = np.array([[480,480,480,480,480,20,20,20,20,20],
                #                                [480,480,480,480,480,20,20,20,20,20],
                #                                [480,480,480,480,480,20,20,20,20,20],
                #                                [480,480,480,480,480,20,20,20,20,20],
                #                                [20,20,20,20,20,480,480,480,480,480],
                #                                [20,20,20,20,20,480,480,480,480,480],
                #                                [20,20,20,20,20,480,480,480,480,480],
                #                                [20,20,20,20,20,480,480,480,480,480]])
                self.label_counts = np.array([[400,400,400,400,400,100,100,100,100,100],
                                               [100,100,100,100,100,400,400,400,400,400]])
                self.label_counts = self.label_counts.tolist()
                self.num_cluster = self.manual_cluster['num-cluster']
                self.list_cut_layers = self.manual_cluster['cut-layers']
                self.infor_cluster = self.manual_cluster['infor-cluster']

                for idx, (client_id, layer_id, performance, cluster, exe_time, net) in enumerate(self.list_clients):
                    if layer_id == 1:
                        self.list_clients[idx] = (client_id, layer_id, performance, cluster, exe_time, net, self.label_counts.pop(), True)
                    else:
                        self.list_clients[idx] = (client_id, layer_id, performance, cluster, exe_time, net, [], True)

        else:
            self.label_counts = self.label_counts.tolist()
            for idx, (client_id, layer_id, performance, cluster, exe_time, net) in enumerate(self.list_clients):
                if layer_id == 1:
                    self.list_clients[idx] = (
                    client_id, layer_id, performance, 0, exe_time, net, self.label_counts.pop(), True)
                else:
                    self.list_clients[idx] = (client_id, layer_id, performance, 0, exe_time, net, [], True)

            self.num_cluster = 1
            self.infor_cluster = [self.total_clients]

            self.list_cut_layers = []
            exe_time_layer_1 = []
            net_layer_1 = []
            exe_time_layer_2 = []
            net_layer_2 = []
            for (client_id, layer_id, performance, cluster, exe_time, net, label, train) in self.list_clients:
                if layer_id == 1:
                    exe_time_layer_1.append(exe_time)
                    net_layer_1.append(net)
                else:
                    exe_time_layer_2.append(exe_time)
                    net_layer_2.append(net)
            cut_point = partition(exe_time_layer_1, net_layer_1, exe_time_layer_2, net_layer_2, self.size_data)
            self.list_cut_layers.append(cut_point)

        self.local_model_parameters = [[[] for _ in range(len(self.total_clients))] for _ in range(self.num_cluster)]
        self.local_client_sizes = [[[] for _ in range(len(self.total_clients))] for _ in range(self.num_cluster)]
        self.local_avg_state_dict = [[] for _ in range(self.num_cluster)]
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

    def avg_all_parameters(self, cluster: int):
        layer_sizes = self.local_client_sizes[cluster]
        layer_params = self.local_model_parameters[cluster]

        for layer_idx, list_state_dicts in enumerate(layer_params):
            list_sizes = layer_sizes[layer_idx]
            if not list_state_dicts or not list_sizes:
                self.local_avg_state_dict[cluster].append({})
                continue
            avg_sd = src.Utils.fedavg_state_dicts(list_state_dicts, weights=list_sizes)
            self.local_avg_state_dict[cluster].append(avg_sd)

    def concatenate_and_avg_clusters(self):
        cluster_state_dicts = []

        for c in range(self.num_cluster):
            avg_layers = self.local_avg_state_dict[c] or []
            if not avg_layers:
                print(f"Warning: cluster {c} has no averaged layers, skipping.")
                continue

            full_dict = {}
            if self.list_cut_layers[c][0] != 0:
                for idx, layer_dict in enumerate(avg_layers):
                    sd = layer_dict
                    if self.model_name != 'ViT' and idx > 0:
                        sd = src.Utils.change_state_dict(layer_dict, self.list_cut_layers[c][idx - 1])
                    full_dict.update(copy.deepcopy(sd))
            else:
                full_dict.update(copy.deepcopy(avg_layers[0]))

            cluster_state_dicts.append(full_dict)

        if not cluster_state_dicts:
            raise RuntimeError("There is no cluster to merge and average.")

        global_state = src.Utils.fedavg_state_dicts(cluster_state_dicts)

        return global_state
