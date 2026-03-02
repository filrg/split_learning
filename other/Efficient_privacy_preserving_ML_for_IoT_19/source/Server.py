import os
import random
import pika
import pickle
import sys
import numpy as np
import copy
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

        self.partition = config["server"]["manual-cluster"]
        # add new attribute
        #self.num_out_cluster = self.partition.get("num-out-cluster", 1)
        #self.num_in_cluster_per_out = self.partition.get("num-in-cluster-per-out", 1)
        self.async_alpha = self.partition.get("async-alpha", 0.6)

        self.model_name = config["server"]["model"]
        self.data_name = config["server"]["data-name"]
        self.total_clients = config["server"]["clients"]
        self.list_cut_layers = config["server"]["manual-cluster"]["cut-layers"]
        self.global_round = config["server"]["global-round"]
        self.local_round = config["server"]["local-round"]
        self.round = self.global_round
        self.validation = config["server"]["validation"]
        
        self.is_clustered = True
        
        #  clustering
        self.out_clusters = {} # {out_idx: [client_ids]}
        self.in_clusters = {} # {out_idx: {in_idx: [client_ids]}}
        self.client_to_hierarchy = {} # {client_id: (out_idx, in_idx)}
        
        self.out_cluster_models = {} # {out_idx: state_dict}
        self.out_cluster_order = [] # Shuffled of out-clusters
        self.current_out_cluster_cursor = 0 # Pointer current out-cluster
        self.current_out_cluster_idx = 0
        self.finished_in_clusters_count = {} # {out_idx: count}
        self.finished_clients_in_cluster = {} # {(out_idx, in_idx): count}

        # Clients
        self.batch_size = config["learning"]["batch-size"]
        self.lr = config["learning"]["learning-rate"]
        self.momentum = config["learning"]["momentum"]
        self.data_distribution = config["server"]["data-distribution"]

        # Data distribution
        self.non_iid = self.data_distribution["non-iid"]
        self.num_label = self.data_distribution["num-label"]
        self.num_sample = self.data_distribution["num-sample"]
        self.random_seed = config["server"]["random-seed"]
        self.label_counts = None
        self.label_ = None

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

        self.channel.basic_qos(prefetch_count=1)
        self.reply_channel = self.connection.channel()
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        debug_mode = config["debug_mode"]
        self.logger = src.Log.Logger(f"{log_path}/app.log", debug_mode)
        src.Log.print_with_color(f"Application start. Server is waiting for {self.total_clients} clients.", "green")
        self.logger.log_info(f"Application start. Server is waiting for {self.total_clients} clients.")

    def distribution(self):
        if self.non_iid:
            # label_distribution = np.random.dirichlet([self.data_distribution["dirichlet"]["alpha"]] * self.num_label,
            #                                          self.total_clients[0])

            label_distribution = np.array(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.09394938, 0.20495232, 0.25764745, 0.20563418, 0.23781668],
                 [0.2824181, 0.132361, 0.09816592, 0.16999675, 0.31705823, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.08073514, 0.13786255, 0.06125086, 0.08391925, 0.04435898, 0.0445482, 0.07578602, 0.18663911,
                  0.20118637, 0.08371351],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.07172973, 0.24979451, 0.28449692, 0.17334935, 0.22062949],
                 [0.25640487, 0.32848751, 0.08951943, 0.24333781, 0.08225038, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.14757221, 0.05964236, 0.06489429, 0.16269761, 0.11871837, 0.0630334, 0.07481413, 0.0249723,
                  0.10654056, 0.17711471],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.35767604, 0.14840493, 0.24900655, 0.06997417, 0.17493831],
                 [0.19780106, 0.31160452, 0.23068388, 0.11227246, 0.14763808, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.12532717, 0.05295416, 0.10434852, 0.07494715, 0.12291418, 0.0860416, 0.08839187, 0.07168553,
                  0.20919395, 0.06419587],
                 ])

            self.label_counts = (label_distribution * self.num_sample).astype(int)
            self.label_ = copy.deepcopy(self.label_counts)
            self.label_ = self.label_.tolist()
        else:
            self.label_counts = np.full((self.total_clients[0], self.num_label), self.num_sample // self.num_label)
            self.label_ = copy.deepcopy(self.label_counts)
            self.label_ = self.label_.tolist()

    def on_request(self, ch, method, props, body):
        message = pickle.loads(body)
        routing_key = props.reply_to
        action = message["action"]
        client_id = message["client_id"]
        layer_id = message["layer_id"]

        self.responses[routing_key] = message

        if action == "REGISTER":
            in_cluster_id = message["in_cluster_id"]
            out_cluster_id = message["out_cluster_id"]
            cid_str = str(client_id)
            if (cid_str, layer_id, in_cluster_id, out_cluster_id) not in self.list_clients:
                self.list_clients.append((cid_str, layer_id, in_cluster_id, out_cluster_id))
                if layer_id == 1:
                    self.edge_device.append((cid_str, layer_id, in_cluster_id, out_cluster_id))
                    
                    # cluster struct
                    if out_cluster_id not in self.out_clusters:
                        self.out_clusters[out_cluster_id] = []
                    self.out_clusters[out_cluster_id].append(cid_str)
                    
                    if out_cluster_id not in self.in_clusters:
                        self.in_clusters[out_cluster_id] = {}
                    if in_cluster_id not in self.in_clusters[out_cluster_id]:
                        self.in_clusters[out_cluster_id][in_cluster_id] = []
                    self.in_clusters[out_cluster_id][in_cluster_id].append(cid_str)
                    
                    self.client_to_hierarchy[cid_str] = (out_cluster_id, in_cluster_id)
                else:
                    self.device_begin.append((cid_str, layer_id, in_cluster_id, out_cluster_id))
                    self.device_stop.append((cid_str, layer_id, in_cluster_id, out_cluster_id))
            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            self.register_clients[layer_id - 1] += 1

            if self.register_clients == self.total_clients:
                self.distribution()
                
                # Initialize Hierarchical Structure
                filepath = f'{self.model_name}_{self.data_name}.pth'
                initial_sd = torch.load(filepath, weights_only=True) if os.path.exists(filepath) else {}
                for o_idx in self.out_clusters.keys():
                    self.out_cluster_models[o_idx] = copy.deepcopy(initial_sd)
                
                # Initialize Shuffled Out-cluster Order
                self.out_cluster_order = list(self.out_clusters.keys())
                random.shuffle(self.out_cluster_order)
                self.current_out_cluster_cursor = 0
                self.current_out_cluster_idx = self.out_cluster_order[0]
                
                src.Log.print_with_color(f"All clients connected. Shuffled Out-cluster order: {self.out_cluster_order}", "green")
                src.Log.print_with_color("Hierarchical structure initialized from predefined IDs.", "green")
                self.logger.log_info(f"Start training round {self.global_round - self.round + 1}")
                self.notify_clients()

        elif action == "NOTIFY":
            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            
            message_pause = {"action": "PAUSE",
                             "message": "Pause training and please send your parameters",
                             "parameters": None}
            
            # Hierarchical Clustering Logic
            # Đợi toàn bộ client trong In-cluster hoàn thành -> pause để thu thập weight
            out_idx, in_idx = self.client_to_hierarchy.get(str(client_id), (0, 0))
            
            # Chỉ xử lý nếu thuộc out-cluster hiện tại
            if out_idx == self.current_out_cluster_idx:
                key = (out_idx, in_idx)
                if key not in self.finished_clients_in_cluster:
                    self.finished_clients_in_cluster[key] = 0
                self.finished_clients_in_cluster[key] += 1
                
                total_in_cluster = len(self.in_clusters[out_idx][in_idx])
                if self.finished_clients_in_cluster[key] == total_in_cluster:
                    src.Log.print_with_color(f">>> In-cluster ({out_idx}, {in_idx}) finished. Requesting parameters for Sync aggregation.", "yellow")
                    for cid in self.in_clusters[out_idx][in_idx]:
                        self.send_to_response(cid, pickle.dumps(message_pause))

        elif action == "UPDATE":
            data_message = message["message"]
            result = message["result"]
            model_state_dict = message["parameters"]
            client_size = message["size"]
            
            src.Log.print_with_color(f"[<<<] Received message from {client_id}: {data_message}", "blue")
            
            # Hierarchical Aggregation (Sync In-cluster -> Async Out-cluster)
            out_idx, in_idx = self.client_to_hierarchy.get(str(client_id), (0, 0))
            
            if out_idx == self.current_out_cluster_idx:
                # Mỗi khi 1 client trong cùng in-cluster gửi update
                self.global_model_parameters[layer_id - 1].append(model_state_dict)
                self.global_client_sizes[layer_id - 1].append(client_size)
                
                total_in_cluster = len(self.in_clusters[out_idx][in_idx])
                # fedavg
                if len(self.global_model_parameters[layer_id - 1]) == total_in_cluster:
                    src.Log.print_with_color(f">>> Sync In-cluster Aggregation ({out_idx}, {in_idx})", "yellow")
                    in_cluster_avg_sd = src.Utils.fedavg_state_dicts(self.global_model_parameters[layer_id - 1], 
                                                                    weights=self.global_client_sizes[layer_id - 1])
                    
                    # Reset
                    self.global_model_parameters[layer_id - 1] = []
                    self.global_client_sizes[layer_id - 1] = []
                    self.finished_clients_in_cluster[(out_idx, in_idx)] = 0
                    
                     # fedasync
                    self.fedasync_aggregate(out_idx, in_cluster_avg_sd)
                    
                    #  Sequential Out-cluster
                    if out_idx not in self.finished_in_clusters_count:
                        self.finished_in_clusters_count[out_idx] = 0
                    self.finished_in_clusters_count[out_idx] += 1
                    
                    total_in_clusters = len(self.in_clusters[out_idx])
                    if self.finished_in_clusters_count[out_idx] == total_in_clusters:
                        src.Log.print_with_color(f">>> Out-cluster {out_idx} training completed.", "green")
                        self.finished_in_clusters_count[out_idx] = 0
                        
                        # chuyển sang outcluster tiếp
                        self.current_out_cluster_cursor += 1
                        
                        if self.current_out_cluster_cursor >= len(self.out_cluster_order):
                            # xong 1 round ( chạy qua hết outcluster)
                            self.round -= 1
                            if self.round <= 0:
                                src.Log.print_with_color(">>> All global rounds completed.", "green")
                                state_dict_full = self.out_cluster_models[out_idx]
                                torch.save(state_dict_full, f'{self.model_name}_{self.data_name}.pth')
                                self.notify_clients(start=False)
                                sys.exit()
                            
                            # next round
                            self.current_out_cluster_cursor = 0
                            random.shuffle(self.out_cluster_order)
                            src.Log.print_with_color(f">>> New Global Round. Shuffled Out-cluster order: {self.out_cluster_order}", "green")

                        # Set next out-cluster
                        next_out_idx = self.out_cluster_order[self.current_out_cluster_cursor]
                        self.out_cluster_models[next_out_idx] = copy.deepcopy(self.out_cluster_models[out_idx])
                        self.current_out_cluster_idx = next_out_idx
                        
                        # Start next Out-cluster
                        src.Log.print_with_color(f">>> Moving to Out-cluster {self.current_out_cluster_idx}", "yellow")
                        self.notify_clients(register=False)

        ch.basic_ack(delivery_tag=method.delivery_tag)

    # cập nhật đồng bộ các incluster trong outcluster (incluster nhanh góp phần nhiều cho weight outcluster)
    def fedasync_aggregate(self, out_idx, in_cluster_sd):
        target_sd = self.out_cluster_models[out_idx]
        alpha = self.async_alpha
        for key in in_cluster_sd.keys():
            if key in target_sd:
                # FedAsync smoothing: W_new = (1-a)W_old + a*W_received
                target_sd[key] = (1.0 - alpha) * target_sd[key].float() + alpha * in_cluster_sd[key].float()
                target_sd[key] = target_sd[key].to(in_cluster_sd[key].dtype)
        src.Log.print_with_color(f">>> FedAsync Out-cluster {out_idx} updated.", "green")



    def notify_clients(self, start=True, register=True, idx=0):
        if not start:
            for node in self.list_clients:
                cid = node[0]
                self.send_to_response(cid, pickle.dumps({"action": "STOP", "message": "Stop!", "parameters": None}))
            return

        if register:
            klass = globals()[f'{self.model_name}_{self.data_name}']
            for node in self.device_begin:
                client_id, layer_id, in_cluster, out_cluster = node[0], node[1], node[2], node[3]
                filepath = f'{self.model_name}_{self.data_name}.pth'
                state_dict = None
                layers = [0, self.list_cut_layers[0]] if layer_id == 1 else [self.list_cut_layers[layer_id-2], self.list_cut_layers[layer_id-1]]
                
                if os.path.exists(filepath):
                    full_sd = torch.load(filepath, weights_only=True)
                    model = klass(end_layer=layers[1]) if layer_id == 1 else klass(start_layer=layers[0], end_layer=layers[1])
                    state_dict = model.state_dict()
                    for key in state_dict.keys(): state_dict[key] = full_sd[key]
                
                label = self.label_.pop() if layer_id == 1 else []
                response = {"action": "START", "message": "Server Accept", "parameters": copy.deepcopy(state_dict),
                            "layers": layers, "model_name": self.model_name, "data_name": self.data_name,
                            "batch_size": self.batch_size, "lr": self.lr, "momentum": self.momentum,
                            "label_count": label, "local_round": self.local_round}
                self.send_to_response(client_id, pickle.dumps(response))
        else:
            # Training Round
            o_idx = self.current_out_cluster_idx
            src.Log.print_with_color(f">>> Training Out-cluster {o_idx}...", "red")
            layers = [0, self.list_cut_layers[0]]
            full_sd = self.out_cluster_models[o_idx]
            
            # Notify all clients in the current out-cluster
            for in_idx, cids in self.in_clusters[o_idx].items():
                for cid in cids:
                    label = self.label_.pop()
                    # Extract client model from full_sd
                    klass = globals()[f'{self.model_name}_{self.data_name}']
                    model = klass(end_layer=layers[1])
                    state_dict = model.state_dict()
                    for key in state_dict.keys(): state_dict[key] = full_sd[key]
                    
                    response = {"action": "START", "parameters": state_dict, 
                                "layers": layers, "model_name": self.model_name, "data_name": self.data_name,
                                "batch_size": self.batch_size, "lr": self.lr, "momentum": self.momentum,
                                "label_count": label, "local_round": self.local_round}
                    self.send_to_response(cid, pickle.dumps(response))

    def start(self):
        self.channel.start_consuming()

    def send_to_response(self, client_id, message):
        reply_queue_name = f'reply_{client_id}'
        self.reply_channel.queue_declare(reply_queue_name, durable=False)
        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
        self.reply_channel.basic_publish(exchange='', routing_key=reply_queue_name, body=message)

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
        full_dict = {}
        for sd in self.avg_state_dict:
            full_dict.update(copy.deepcopy(sd))
        return full_dict
