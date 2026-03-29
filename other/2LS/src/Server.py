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
from src.val.get_val import get_val
from src.model.BERT_AGNEWS import BERT_AGNEWS
from src.model.KWT_SPEECHCOMMANDS import KWT_SPEECHCOMMANDS
from src.model.VGG16_CIFAR10 import VGG16_CIFAR10

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
        self.num_cluster = config["server"]["num-cluster"]
        self.cut_layer = config["server"]["cut-layer"]
        self.info_cluster = config["server"]["info-cluster"]
        self.global_round = config["server"]["global-round"]
        self.round = self.global_round
        self.global_model = None

        # Clients
        self.learning = config["learning"]
        self.data_distribution = config["server"]["data-distribution"]

        # Data distribution
        self.non_iid = self.data_distribution["non-iid"]
        self.num_label = self.data_distribution["num-label"]
        self.num_sample = self.data_distribution["num-sample"]
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

        self.out_cluster_ids = list(range(self.num_cluster))
        self.count_notify = [] # list
        self.count_update = [] # list
        self.check_in_cluster = []
        self.in_params = None # list([[],[]])
        self.in_sizes = None # list([[],[]])
        self.register_clients = [0 for _ in range(len(self.total_clients))]
        self.responses = {}  # Save response
        self.list_clients = []
        self.round_result = True

        self.current_out_cluster = 0
        self.full_state_dict = None

        self.channel.basic_qos(prefetch_count=1)
        self.reply_channel = self.connection.channel()
        self.channel.basic_consume(queue='rpc_queue', on_message_callback=self.on_request)

        debug_mode = config["debug_mode"]

        self.logger = src.Log.Logger(f"{log_path}/app.log", debug_mode)

        self.logger.log_info(f"Application start. Server is waiting for {self.total_clients} clients.")

    def distribution(self):
        if self.non_iid:
            if self.data_name == "AGNEWS":
                label_distribution = np.array([[0.75, 0.0, 0.0, 0.25],
                                               [0.75, 0.0, 0.0, 0.25],
                                               [0.75, 0.0, 0.0, 0.25],
                                               [0.0, 0.75, 0.0, 0.25],
                                               [0.0, 0.75, 0.0, 0.25],
                                               [0.0, 0.75, 0.0, 0.25],
                                               [0.0, 0.0, 0.75, 0.25],
                                               [0.0, 0.0, 0.75, 0.25],
                                               [0.0, 0.0, 0.75, 0.25],
                                               ])
            else:
                 label_distribution = np.array([[0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                                               [0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                                               [0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
                                               [0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.1],
                                               [0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.1],
                                               [0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0, 0.1],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.1],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.1],
                                               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.1],
                                               ])

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
            in_cluster_id = message["in_cluster_id"]
            out_cluster_id = message["out_cluster_id"]
            idx = message["idx"]

            if (client_id, layer_id, in_cluster_id, out_cluster_id, idx) not in self.list_clients:
                self.list_clients.append((client_id, layer_id, in_cluster_id, out_cluster_id, idx))
            
            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            self.register_clients[layer_id - 1] += 1

            if self.register_clients == self.total_clients:
                self.distribution()
                self.set_up()

                self.logger.log_info(f"Start training round {self.global_round - self.round + 1}")
                src.Log.print_with_color(f"Start training round {self.global_round - self.round + 1}", "yellow")

                self.count_notify = copy.deepcopy(self.info_cluster[0])
                self.count_update = [x * 2 for x in self.info_cluster[0]]
                self.in_params = [[[],[]] for _ in range(len(self.info_cluster[0]))]
                self.in_sizes = [[[],[]] for _ in range(len(self.info_cluster[0]))]

                src.Log.print_with_color(f"List out-cluster : {self.out_cluster_ids}", "yellow")
                self.current_out_cluster = self.out_cluster_ids.pop(0)
                src.Log.print_with_color(f"Start out-cluster {self.current_out_cluster}", "yellow")
                self.notify_clients()

        elif action == "NOTIFY":
            src.Log.print_with_color(f"[<<<] Received message from client: {message}", "blue")
            in_cluster = message["in_cluster_id"]

            message = {"action": "PAUSE",
                        "message": "Pause training and please send your parameters"}

            self.count_notify[in_cluster] -= 1
            if self.count_notify[in_cluster] == 0:
                src.Log.print_with_color(f"Received finish training notification clients from in cluster {in_cluster}.", "yellow")
                for (client_id, layer_id, in_cluster_id , out_cluster_id, _, _) in self.list_clients:
                    if (out_cluster_id == self.current_out_cluster or layer_id == 2) and (in_cluster_id == in_cluster):
                        self.send_to_response(client_id, pickle.dumps(message))

        elif action == "UPDATE":
            data_message = message["message"]
            result = message["result"]
            model_state_dict = message["parameters"]
            client_size = message["size"]
            in_cluster = message["in_cluster_id"]

            if not result:
                self.round_result = False
            src.Log.print_with_color(f"[<<<] Received message from {client_id}: {data_message}", "blue")

            self.count_update[in_cluster] -= 1
            self.in_params[in_cluster][layer_id - 1].append(model_state_dict)
            self.in_sizes[in_cluster][layer_id - 1].append(client_size)

            if self.count_update[in_cluster] == 0:
                self.check_in_cluster.append(in_cluster)

            if len(self.check_in_cluster) == len(self.info_cluster[self.current_out_cluster]):
                avg_in_cluster = self.avg_in_clusters()

                for num, check in enumerate(self.check_in_cluster):
                    alpha =  float(1 / (1 + num))
                    self.global_model = self.fed_async_aggregate(self.global_model, avg_in_cluster[check], alpha)
                    torch.save(self.global_model, f'{self.model_name}_{self.data_name}.pth')

                if len(self.out_cluster_ids) == 0:
                    if self.round_result:
                        # Test
                        if not get_val(self.model_name, self.data_name, self.global_model, self.logger):
                            self.logger.log_warning("Training failed!")
                            src.Log.print_with_color("Training failed!", "yellow")
                            self.round = 0
                        else:
                            self.round -= 1
                    else:
                        self.round = 0

                    if self.round > 0:
                        self.logger.log_info(f"Start training round {self.global_round - self.round + 1}")

                        self.out_cluster_ids = list(range(self.num_cluster))
                        random.shuffle(self.out_cluster_ids)
                        self.global_model = None
                        self.reset()

                        src.Log.print_with_color(f"List out-cluster : {self.out_cluster_ids}", "yellow")
                        self.current_out_cluster = self.out_cluster_ids.pop(0)
                        self.notify_clients(True, self.current_out_cluster)
                        src.Log.print_with_color(f"Start out-cluster {self.current_out_cluster}", "yellow")
                    else:
                        self.logger.log_info("Stop training !!!")
                        self.notify_clients(start=False)
                        sys.exit()
                else:
                    self.reset()

                    self.current_out_cluster = self.out_cluster_ids.pop(0)
                    src.Log.print_with_color(f"Start out-cluster {self.current_out_cluster}", "yellow")
                    self.notify_clients(True, self.current_out_cluster)


        ch.basic_ack(delivery_tag=method.delivery_tag)

    def fed_async_aggregate(self, out_cluster_sd, in_cluster_sd, alpha=1.0):
        if out_cluster_sd is None:
            out_cluster_sd = in_cluster_sd
            src.Log.print_with_color(f">>> FedAsync Out-cluster {self.current_out_cluster} updated (alpha={alpha}).", "green")
        else:
            for key in in_cluster_sd.keys():
                out_cluster_sd[key] = (1.0 - alpha) * out_cluster_sd[key].float() + alpha * in_cluster_sd[key].float()
                out_cluster_sd[key] = out_cluster_sd[key].to(in_cluster_sd[key].dtype)
            src.Log.print_with_color(f">>> FedAsync Out-cluster {self.current_out_cluster} updated (alpha={alpha}).", "green")
        return out_cluster_sd

    def notify_clients(self, start=True, out_id=0):
        filepath = f'{self.model_name}_{self.data_name}.pth'

        for (client_id, layer_id, _, out_cluster_id, _, labels) in self.list_clients:
            state_dict = None
            if start:
                if out_cluster_id == out_id or out_cluster_id == -1:
                    if os.path.exists(filepath):
                        self.full_state_dict = torch.load(filepath, weights_only=True)

                        if self.model_name == 'VGG16':
                            klass = VGG16_CIFAR10
                        elif self.model_name == "KWT":
                            klass = KWT_SPEECHCOMMANDS
                        else:
                            klass = BERT_AGNEWS

                        if layer_id == 1:
                            model = klass(end_layer=self.cut_layer)
                        else:
                            model = klass(start_layer=self.cut_layer)
                        state_dict = model.state_dict()
                        keys = state_dict.keys()

                        for key in keys:
                            state_dict[key] = self.full_state_dict[key]

                        src.Log.print_with_color(f"Load model successfully", "green")
                    else:
                        src.Log.print_with_color(f"File {filepath} does not exist.", "yellow")

                    src.Log.print_with_color(f"[>>>] Sent start training request to client {client_id}", "red")

                    response = {"action": "START",
                                "message": "Server accept the connection!",
                                "parameters": state_dict,
                                "cut_layer": self.cut_layer,
                                "model_name": self.model_name,
                                "data_name": self.data_name,
                                "learning": self.learning,
                                "label_count": labels}
                    self.send_to_response(client_id, pickle.dumps(response))
                else:
                    continue
            else:
                src.Log.print_with_color(f"[>>>] Sent stop training request to client {client_id}", "red")
                response = {"action": "STOP",
                            "message": "Stop training!"}
                self.send_to_response(client_id, pickle.dumps(response))

    def set_up(self):
        self.label_counts = self.label_counts.tolist()
        new_list_client = []
        for (client_id, layer_id, in_cluster_id, out_cluster_id, idx) in self.list_clients:
            if layer_id == 1:
                new_list_client.append((client_id, layer_id, in_cluster_id, out_cluster_id, idx, self.label_counts.pop(0)))
            else:
                new_list_client.append((client_id, layer_id, in_cluster_id, out_cluster_id, idx, []))

        self.list_clients = new_list_client

    def start(self):
        self.channel.start_consuming()

    def send_to_response(self, client_id, message):
        reply_queue_name = f'reply_{client_id}'
        self.reply_channel.queue_declare(reply_queue_name, durable=False)
        src.Log.print_with_color(f"[>>>] Sent notification to client {client_id}", "red")
        self.reply_channel.basic_publish(exchange='', routing_key=reply_queue_name, body=message)

    def avg_in_clusters(self):
        avg_in_cluster = []

        for i in range(len(self.in_params)):
            list_params = self.in_params[i]
            list_sizes =  self.in_sizes[i]
            full_dict = {}

            for idx, layer_dict in enumerate(list_params):
                sd = src.Utils.fedavg_state_dicts(layer_dict, list_sizes[idx])
                full_dict.update(copy.deepcopy(sd))

            avg_in_cluster.append(full_dict)

        return avg_in_cluster

    def reset(self):
        self.check_in_cluster = []
        self.count_notify = copy.deepcopy(self.info_cluster[0])
        self.count_update = [x * 2 for x in self.info_cluster[0]]
        self.in_params = [[[], []] for _ in range(len(self.info_cluster[0]))]
        self.in_sizes = [[[], []] for _ in range(len(self.info_cluster[0]))]
