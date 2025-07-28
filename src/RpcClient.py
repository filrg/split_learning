import time
import pickle
import pika
import copy
import torchvision

from collections import defaultdict
from tqdm import tqdm

import src.Log
import src.Model
from src.model import *
from src.Attack import *


class RpcClient:
    def __init__(self, client_id, address, username, password, train_func, device, args):
        self.client_id = client_id
        self.layer_id = args.layer_id
        self.address = address
        self.username = username
        self.password = password
        self.train_func = train_func
        self.device = device
        self.args = args

        self.channel = None
        self.connection = None
        self.response = None
        self.model = None
        self.global_model = None
        self.cluster = None
        self.label_count = None
        self.connect()

        self.train_set = None
        self.label_to_indices = None
        self.round_count = 0
        self.mapping = parse_mapping(self.args.label_mapping) if self.args.label_mapping else {}

    def wait_response(self):
        status = True
        reply_queue_name = f'reply_{self.client_id}'
        self.channel.queue_declare(reply_queue_name, durable=False)
        while status:
            method_frame, header_frame, body = self.channel.basic_get(queue=reply_queue_name, auto_ack=True)
            if body:
                status = self.response_message(body)
            time.sleep(0.5)

    def response_message(self, body):
        self.response = pickle.loads(body)
        src.Log.print_with_color(f"[<<<] Client received: {self.response['message']}", "blue")
        action = self.response["action"]
        state_dict = self.response["parameters"]

        if action == "START":
            special = self.response["special"]
            model_name = self.response["model_name"]
            cut_layers = self.response['layers']
            label_count = self.response['label_count']
            num_layers = self.response['num_layers']
            clip_grad_norm = self.response['clip_grad_norm']
            data_name = self.response["data_name"]

            if self.label_count is None:
                self.label_count = label_count
            if self.response['cluster'] is not None:
                self.cluster = self.response['cluster']
            if self.label_count is not None:
                src.Log.print_with_color(f"Label distribution of client: {self.label_count}", "yellow")

            # Load training dataset
            if self.layer_id == 1:
                if not self.train_set and not self.label_to_indices:
                    self.load_dataset(data_name)
                elif self.round_count == self.args.attack_round:
                    self.load_dataset(data_name)

            # Load model
            if self.model is None:
                if 'MNIST' in data_name:
                    klass = globals()[f'{model_name}_MNIST']
                else:
                    klass = globals()[f'{model_name}_{data_name}']

                if model_name != 'ViT':
                    full_model = klass()
                    if cut_layers[1] != 0:
                        from_layer = cut_layers[0]
                        to_layer = cut_layers[1]
                        if to_layer == -1:
                            self.model = nn.Sequential(*nn.ModuleList(full_model.children())[from_layer:])
                        else:
                            self.model = nn.Sequential(*nn.ModuleList(full_model.children())[from_layer:to_layer])
                    else:
                        self.model = nn.Sequential(*nn.ModuleList(full_model.children())[:])

                else:
                    if cut_layers[1] != 0:
                        if cut_layers[1] == -1:
                            self.model = klass(start_layer=cut_layers[0])
                        else:
                            self.model = klass(start_layer=cut_layers[0], end_layer=cut_layers[1])
                    else:
                        self.model = klass()
                self.model.to(self.device)
            batch_size = self.response["batch_size"]
            lr = self.response["lr"]
            momentum = self.response["momentum"]
            compute_loss = self.response["compute_loss"]
            control_count = self.response["control_count"]

            # Read parameters and load to model
            if state_dict:
                self.model.load_state_dict(state_dict)
            if self.response["cluster"] is not None and compute_loss["mode"] != 'normal':
                self.global_model = copy.copy(self.model)

            # Start training
            if self.layer_id == 1:
                selected_indices = []
                for label, count in enumerate(self.label_count):
                    selected_indices.extend(random.sample(self.label_to_indices[label], count))

                subset = torch.utils.data.Subset(self.train_set, selected_indices)
                train_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
                if cut_layers[1] != 0:
                    result, size = self.train_func(self.model, self.global_model, self.label_count, lr, momentum, clip_grad_norm, compute_loss, num_layers, control_count, train_loader, self.cluster, special, alone_train=False)
                else:
                    result, size = self.train_func(self.model, self.global_model, self.label_count, lr, momentum, clip_grad_norm, compute_loss, num_layers, control_count, train_loader, self.cluster, special, alone_train=True)
            else:
                result, size = self.train_func(self.model, self.global_model, self.label_count, lr, momentum, clip_grad_norm, compute_loss, num_layers, control_count, None, self.cluster, special)

            # Stop training, then send parameters to server
            model_state_dict = copy.deepcopy(self.model.state_dict())
            if self.device != "cpu":
                for key in model_state_dict:
                    model_state_dict[key] = model_state_dict[key].to('cpu')
            data = {"action": "UPDATE", "client_id": self.client_id, "layer_id": self.layer_id,
                    "result": result, "size": size, "cluster": self.cluster,
                    "message": "Sent parameters to Server", "parameters": model_state_dict}
            src.Log.print_with_color("[>>>] Client sent parameters to server", "red")
            self.send_to_server(data)
            self.round_count += 1
            return True
        elif action == "STOP":
            return False

    def connect(self):
        credentials = pika.PlainCredentials(self.username, self.password)
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.address, 5672, '/', credentials))
        self.channel = self.connection.channel()

    def send_to_server(self, message):
        self.connect()
        self.response = None

        self.channel.queue_declare('rpc_queue', durable=False)
        self.channel.basic_publish(exchange='',
                                   routing_key='rpc_queue',
                                   body=pickle.dumps(message))

        return self.response

    def load_dataset(self, data_name):
        if data_name == "MNIST":
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            self.train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                                        transform=transform_train)
        elif data_name == "FASHION_MNIST":
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            self.train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True,
                                                               transform=transform_train)
        elif data_name == "CIFAR10":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            if not self.train_set:
                self.train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                              transform=transform_train)
                self.label_to_indices = defaultdict(list)
                for idx, (_, label) in tqdm(enumerate(self.train_set)):
                    self.label_to_indices[int(label)].append(idx)

            if self.round_count == self.args.attack_round:
                if self.args.attack_mode == "normal":
                    pass
                elif self.args.attack_mode == "pixel":
                    src.Log.print_with_color("Start pixel attack", "red")
                    self.train_set = BackdoorCIFAR10(root='./data', train=True, transform=transform_train,
                                                     poison_rate=self.args.poison_rate,
                                                     trigger_size=self.args.trigger_size,
                                                     trigger_location=self.args.trigger_location,
                                                     trigger_color=tuple(self.args.trigger_color),
                                                     label_mapping=self.mapping)
                elif self.args.attack_mode == "semantic":
                    src.Log.print_with_color("Start semantic attack", "red")
                    self.train_set = SemanticBackdoorCIFAR10(root='./data', train=True, transform=transform_train,
                                                             poison_rate=self.args.poison_rate,
                                                             stripe_width=self.args.stripe_width,
                                                             alpha=self.args.alpha,
                                                             stripe_orientation=self.args.stripe_orientation,
                                                             label_mapping=self.mapping)
                else:
                    raise ValueError(f"Attack mode '{self.args.attack_mode}' is not valid.")

        else:
            raise ValueError(f"Data name '{data_name}' is not valid.")


