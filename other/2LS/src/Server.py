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
        self.out_cluster_models = {}  # {out_idx: state_dict}
        self.out_cluster_order = []  # Shuffled of out-clusters
        self.current_out_cluster_cursor = 0  # Pointer current out-cluster
        self.current_out_cluster_idx = 0
        self.finished_clients_in_cluster = {}  # {(out_idx, in_idx): count}
        self.finished_upper_clients_count = {} # {out_idx: count} — Phase 2 only

        # FedAsync: track thứ tự in-cluster đến cho mỗi out-cluster
        self.incluster_fedasync_order = {}  # {out_idx: [in_idx_first, in_idx_second, ...]}
        self.incluster_l1_avg = {}  # {(out_idx, in_idx): state_dict} — saved L1 FedAvg result
        self.incluster_l2_finished = {}  # {(out_idx, in_idx): count} — L2 done per in-cluster

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

        # Model (Isolated buffers per cluster: (layer, out_idx, in_idx))
        self.global_model_parameters = {}
        self.global_client_sizes = {}

        # Sequential
        self.edge_device = []
        self.device_begin = []
        self.device_stop = []
        self.avg_state_dict = []
        self.upper_clients = {}  # {out_cluster_id: [(cid, layer_id)]} for layer 2 clients

        self.channel.basic_qos(prefetch_count=1)
        self.reply_channel = self.connection.channel()
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        debug_mode = config["debug_mode"]
        self.logger = src.Log.Logger(f"{log_path}/app.log", debug_mode)
        src.Log.print_with_color(f"Application start. Server is waiting for {self.total_clients} clients.", "green")
        self.logger.log_info(f"Application start. Server is waiting for {self.total_clients} clients.")


    def distribution(self):
        if self.non_iid:
            label_distribution = np.array([[0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                                           [0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.1],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.1],
                                           [0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                                           [0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.1],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.1],
                                           [0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                                           [0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.1],
                                           [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.1],
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
        layer_id = int(message["layer_id"])

        self.responses[routing_key] = message
        ch.basic_ack(delivery_tag=method.delivery_tag)  # Ack immediately — always

        if action == "REGISTER":
            in_cluster_id = message["in_cluster_id"]
            out_cluster_id = message["out_cluster_id"]
            idx = message.get("idx", -1)
            cid_str = str(client_id)
            if (cid_str, layer_id, in_cluster_id, out_cluster_id, idx) not in self.list_clients:
                self.list_clients.append((cid_str, layer_id, in_cluster_id, out_cluster_id, idx))
            
            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            self.register_clients[layer_id - 1] += 1

            if self.register_clients == self.total_clients:
                self.distribution()

                filepath = f'{self.model_name}_{self.data_name}.pth'
                initial_sd = torch.load(filepath, weights_only=True) if os.path.exists(filepath) else {}
                
                # node[3] là out_cluster_id
                unique_out_clusters = sorted(list(set(node[3] for node in self.list_clients))) # thứ tự 0 1 2
                for o_idx in unique_out_clusters:
                    self.out_cluster_models[o_idx] = copy.deepcopy(initial_sd)

                # Initialize Shuffled Out-cluster Order
                self.out_cluster_order = unique_out_clusters
                random.shuffle(self.out_cluster_order)
                self.current_out_cluster_cursor = 0
                self.current_out_cluster_idx = self.out_cluster_order[0]

                src.Log.print_with_color(f"All clients connected. Shuffled Out-cluster order: {self.out_cluster_order}",
                                         "green")
                src.Log.print_with_color("Hierarchical structure initialized from predefined IDs.", "green")
                self.logger.log_info(f"Start training round {self.global_round - self.round + 1}")
                self.notify_clients(register=False)

        elif action == "NOTIFY":
            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")

            message_pause = {"action": "PAUSE",
                             "message": "Pause training and please send your parameters",
                             "parameters": None}

            node = next((n for n in self.list_clients if n[0] == str(client_id)), None)
            if int(layer_id) > 1:
                out_idx = self.current_out_cluster_idx
                in_idx = node[2] if node else 0
            elif node:
                out_idx, in_idx = node[3], node[2]
            else:
                out_idx, in_idx = 0, 0

            if out_idx == self.current_out_cluster_idx:
                key = (out_idx, in_idx)
                if key not in self.finished_clients_in_cluster:
                    self.finished_clients_in_cluster[key] = 0
                self.finished_clients_in_cluster[key] += 1

                clients_in_cluster = [n[0] for n in self.list_clients if n[3] == out_idx and n[2] == in_idx and n[1] == 1]
                if self.finished_clients_in_cluster[key] == len(clients_in_cluster):
                    src.Log.print_with_color(f">>> In-cluster ({out_idx}, {in_idx}) finished. Requesting parameters.", "yellow")
                    for cid in clients_in_cluster:
                        self.send_to_response(cid, pickle.dumps(message_pause))

        elif action == "UPDATE":
            data_message = message["message"]
            result = message["result"]
            model_state_dict = message["parameters"]
            client_size = message["size"]

            src.Log.print_with_color(f"[<<<] Received message from {client_id}: {data_message}", "blue")

            node = next((n for n in self.list_clients if n[0] == str(client_id)), None)
            if layer_id > 1:
                out_idx = self.current_out_cluster_idx
                in_idx = node[2] if node else 0
                #src.Log.print_with_color(f">>> Mapping Split Server {client_id} to active Out-cluster {out_idx}", "yellow")
            elif node:
                out_idx, in_idx = node[3], node[2]
            else:
                out_idx, in_idx = 0, 0

            key = (out_idx, in_idx)
            cid_str = str(client_id)

            if layer_id == 1:
                if out_idx == self.current_out_cluster_idx:
                    cluster_key = (layer_id, out_idx, in_idx)
                    if cluster_key not in self.global_model_parameters:
                        self.global_model_parameters[cluster_key] = []
                        self.global_client_sizes[cluster_key] = []

                    self.global_model_parameters[cluster_key].append(model_state_dict)
                    self.global_client_sizes[cluster_key].append(client_size)

                    # số client ở layer 1 của cur_o_idx
                    clients_in_cluster = [n[0] for n in self.list_clients if int(n[3]) == out_idx and int(n[2]) == in_idx and int(n[1]) == 1]
                    total_in_cluster = len(clients_in_cluster)

                    # FedAvg khi in-cluster đã nhận đủ update từ tất cả client
                    if len(self.global_model_parameters[cluster_key]) == total_in_cluster:
                        src.Log.print_with_color(f">>> Sync In-cluster L1 FedAvg ({out_idx}, {in_idx})", "yellow")
                        in_cluster_avg_sd = src.Utils.fedavg_state_dicts(self.global_model_parameters[cluster_key],
                                                                         weights=self.global_client_sizes[cluster_key])
                        self.global_model_parameters[cluster_key] = []
                        self.global_client_sizes[cluster_key] = []
                        self.finished_clients_in_cluster[(out_idx, in_idx)] = 0

                        # Track thứ tự in-cluster đến
                        if out_idx not in self.incluster_fedasync_order:
                            self.incluster_fedasync_order[out_idx] = []
                        order_list = self.incluster_fedasync_order[out_idx]
                        order_list.append(in_idx)

                        alpha = 1.0 if len(order_list) == 1 else 0.5
                        src.Log.print_with_color(f">>> In-cluster ({out_idx}, {in_idx}) arrived {'FIRST' if alpha == 1.0 else 'LATER'}. alpha={alpha}", "green")

                        # Lưu L1 FedAvg result — CHƯA FedAsync, chờ L2
                        self.incluster_l1_avg[(out_idx, in_idx)] = in_cluster_avg_sd

                        # Pause L1 clients
                        message_pause_l1 = {"action": "PAUSE", "message": "In-cluster done. Waiting.", "parameters": None}
                        for cid in clients_in_cluster:
                            self.send_to_response(cid, pickle.dumps(message_pause_l1))

                        # Gửi PAUSE cho L2 clients thuộc in-cluster này
                        l2_clients_this_ic = [n for n in self.list_clients if int(n[1]) > 1 and int(n[2]) == in_idx]

                        if l2_clients_this_ic:
                            message_pause_l2 = {"action": "PAUSE", "message": f"Send L2 parameters for in-cluster {in_idx}.", "parameters": None}
                            seen = set()
                            for node in reversed(l2_clients_this_ic):
                                role_key = (node[1], node[2], node[3], node[4])
                                if role_key not in seen:
                                    seen.add(role_key)
                                    src.Log.print_with_color(f">>> Sending PAUSE to L2 client {node[0]} for in-cluster {in_idx}", "yellow")
                                    self.send_to_response(node[0], pickle.dumps(message_pause_l2))
                        else:
                            # Không có L2 → FedAsync L1 trực tiếp (paper: model chỉ có L1)
                            src.Log.print_with_color(f">>> No L2 for in-cluster ({out_idx}, {in_idx}). FedAsync L1 only, alpha={alpha}", "green")
                            self.fedasync_aggregate(out_idx, in_cluster_avg_sd, alpha=alpha)
                            self.incluster_l1_avg.pop((out_idx, in_idx), None)
                            self.check_out_cluster_completion(out_idx)

                else:
                    message_pause = {"action": "PAUSE", "message": "Round mismatch. Waiting...", "parameters": None}
                    self.send_to_response(cid_str, pickle.dumps(message_pause))

            elif layer_id > 1:
                # Accumulate L2 update per in-cluster
                src.Log.print_with_color(f">>> Received UPDATE from Upper Layer client {client_id} (Layer {layer_id}, in-cluster {in_idx})", "yellow")
                l2_key = (layer_id, out_idx, in_idx)
                if l2_key not in self.global_model_parameters:
                    self.global_model_parameters[l2_key] = []
                    self.global_client_sizes[l2_key] = []
                self.global_model_parameters[l2_key].append(model_state_dict)
                self.global_client_sizes[l2_key].append(client_size)

                # Đếm L2 per in-cluster
                l2_ic_key = (out_idx, in_idx)
                if l2_ic_key not in self.incluster_l2_finished:
                    self.incluster_l2_finished[l2_ic_key] = 0
                self.incluster_l2_finished[l2_ic_key] += 1

                # Số L2 clients (unique roles) thuộc in-cluster này
                total_upper_this_ic = len(set((n[1], n[2], n[3], n[4]) for n in self.list_clients if int(n[1]) > 1 and int(n[2]) == in_idx))

                if self.incluster_l2_finished[l2_ic_key] >= total_upper_this_ic:
                    # FedAvg L2
                    avg_sd_l2 = src.Utils.fedavg_state_dicts(self.global_model_parameters[l2_key],
                                                              weights=self.global_client_sizes[l2_key])
                    self.global_model_parameters[l2_key] = []
                    self.global_client_sizes[l2_key] = []

                    # Merge L1 avg + L2 avg → full model
                    l1_avg = self.incluster_l1_avg.pop(l2_ic_key, {})
                    merged_sd = {}
                    merged_sd.update(l1_avg)
                    merged_sd.update(avg_sd_l2)

                    # Lấy alpha theo thứ tự in-cluster đến
                    order_list = self.incluster_fedasync_order.get(out_idx, [])
                    arrival_pos = order_list.index(in_idx) if in_idx in order_list else 0
                    alpha = 1.0 if arrival_pos == 0 else 0.5

                    src.Log.print_with_color(f">>> In-cluster ({out_idx}, {in_idx}) L1+L2 merged. FedAsync alpha={alpha} (paper Alg.1)", "green")
                    self.fedasync_aggregate(out_idx, merged_sd, alpha=alpha)

                    self.check_out_cluster_completion(out_idx)

    def check_out_cluster_completion(self, out_idx):
        # Kiểm tra tất cả in-cluster đã FedAsync xong
        all_l1_in_oc = set(int(n[2]) for n in self.list_clients if int(n[1]) == 1 and int(n[3]) == out_idx)
        order_list = self.incluster_fedasync_order.get(out_idx, [])
        is_done = len(order_list) >= len(all_l1_in_oc) if all_l1_in_oc else True
        
        # Kiểm tra L2 per in-cluster: tất cả in-cluster phải hoàn thành L2
        all_l2_done = True
        for ic_idx in all_l1_in_oc:
            l2_count_for_ic = len(set((n[1], n[2], n[3], n[4]) for n in self.list_clients if int(n[1]) > 1 and int(n[2]) == ic_idx))
            if l2_count_for_ic > 0:
                finished_l2_ic = self.incluster_l2_finished.get((out_idx, ic_idx), 0)
                if finished_l2_ic < l2_count_for_ic:
                    all_l2_done = False
                    break

        if is_done and all_l2_done:
            src.Log.print_with_color(f">>> Out-cluster {out_idx} FULLY completed (L1 & L2+).", "green")
            
            # Validation: tổng hợp mô hình và in accuracy
            state_dict_full = self.out_cluster_models[out_idx]
            if len(state_dict_full) > 0:
                src.Log.print_with_color(f">>> Running validation for Out-cluster {out_idx}...", "yellow")
                src.Validation.test(self.model_name, self.data_name, state_dict_full, self.logger)

            # Reset cho out-cluster này
            self.incluster_fedasync_order[out_idx] = []
            self.finished_upper_clients_count[out_idx] = 0
            # Reset L2 per in-cluster counters
            for ic_idx in all_l1_in_oc:
                self.incluster_l2_finished.pop((out_idx, ic_idx), None)
                self.incluster_l1_avg.pop((out_idx, ic_idx), None)
            
            # chuyển sang outcluster tiếp
            self.current_out_cluster_cursor += 1

            if self.current_out_cluster_cursor >= len(self.out_cluster_order):
                # xong 1 round ( chạy qua hết outcluster)
                self.round -= 1
                if self.round <= 0:
                    src.Log.print_with_color(">>> All global rounds completed.", "green")
                    state_dict_full = self.out_cluster_models[out_idx]
                    torch.save(state_dict_full, f'{self.model_name}_{self.data_name}.pth')
                    src.Log.print_with_color(">>> Server training process total completion.", "green")
                    return

                # next round
                self.current_out_cluster_cursor = 0
                random.shuffle(self.out_cluster_order)
                src.Log.print_with_color(
                    f">>> New Global Round. Shuffled Out-cluster order: {self.out_cluster_order}", "green")

            # Set next out-cluster
            next_out_idx = self.out_cluster_order[self.current_out_cluster_cursor]
            self.out_cluster_models[next_out_idx] = copy.deepcopy(self.out_cluster_models[out_idx])
            self.current_out_cluster_idx = next_out_idx

            # Start next Out-cluster
            src.Log.print_with_color(f">>> Moving to Out-cluster {self.current_out_cluster_idx}", "yellow")
            self.notify_clients(register=False)

    # FedAsync: W_new = (1-alpha)*W_old + alpha*W_received
    def fedasync_aggregate(self, out_idx, in_cluster_sd, alpha=1.0):
        target_sd = self.out_cluster_models[out_idx]
        for key in in_cluster_sd.keys():
            if key in target_sd:
                target_sd[key] = (1.0 - alpha) * target_sd[key].float() + alpha * in_cluster_sd[key].float()
                target_sd[key] = target_sd[key].to(in_cluster_sd[key].dtype)
            else:
                # Key chưa tồn tại (model khởi tạo rỗng) → thêm trực tiếp
                target_sd[key] = in_cluster_sd[key].clone()
        src.Log.print_with_color(f">>> FedAsync Out-cluster {out_idx} updated (alpha={alpha}).", "green")

    def notify_clients(self, start=True, register=True):
        if not start:
            for node in self.list_clients:
                self.send_to_response(node[0], pickle.dumps({"action": "STOP", "message": "Stop!", "parameters": None}))
            return

        if register:
            pass
        else:
            o_idx = self.current_out_cluster_idx
            full_sd = self.out_cluster_models[o_idx]
            cut = int(self.list_cut_layers[0][0] if isinstance(self.list_cut_layers[0], list) else self.list_cut_layers[0])
            klass = globals()[f'{self.model_name}_{self.data_name}']

            src.Log.print_with_color(f">>> Starting Training Round for Out-cluster {o_idx}...", "red")

            # Notify cho những thằng ở o_idx hiện tại thoi
            for role in set(n[1:] for n in self.list_clients):
                layer_id, in_idx, out_idx, idx = role
                if not ((layer_id == 1 and out_idx == o_idx) or layer_id > 1):
                    continue

                client_id = next(n[0] for n in reversed(self.list_clients) if n[1:] == role)
                layers = [0, cut] if layer_id == 1 else [cut, -1]
                model = klass(end_layer=cut) if layer_id == 1 else klass(start_layer=cut)

                state_dict = model.state_dict()
                if len(full_sd) > 0:
                    for key in state_dict.keys():
                        if key in full_sd:
                            state_dict[key] = full_sd[key]

                label = self.label_[idx] if (layer_id == 1 and self.label_) else []

                response = {"action": "START", "message": "Training Start", "parameters": state_dict,
                            "layers": layers, "model_name": self.model_name, "data_name": self.data_name,
                            "batch_size": self.batch_size, "lr": self.lr, "momentum": self.momentum,
                            "label_count": label, "local_round": self.local_round, "cluster": in_idx,
                            "out_cluster_id": o_idx}

                src.Log.print_with_color(f">>> Notifying client {client_id} (layer {layer_id}) for out-cluster {o_idx}, in-cluster {in_idx}", "yellow")
                self.send_to_response(client_id, pickle.dumps(response))

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