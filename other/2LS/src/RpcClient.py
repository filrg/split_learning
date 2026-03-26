import time
import pickle
import random
import copy
import torchvision
import torchvision.transforms as transforms
import torch

from collections import defaultdict
from tqdm import tqdm

import src.Log
from src.model import *

from peft import LoraConfig, get_peft_model


class RpcClient:
    def __init__(self, client_id, layer_id, channel, train_func, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.train_func = train_func
        self.device = device

        self.response = None
        self.model = None
        self.label_count = None
        self.peft_config = None

        self.train_set = None
        self.label_to_indices = None

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
        src.Log.print_with_color(f"[<<<] Client received: {self.response.get('message', 'No message')}", "blue")
        action = self.response["action"]
        state_dict = self.response["parameters"]

        if action == "START":
            model_name = self.response["model_name"]
            cut_layers = self.response['layers']
            label_count = self.response['label_count']
            data_name = self.response["data_name"]
            local_round = self.response["local_round"]

            if self.label_count is None:
                self.label_count = label_count

            if self.label_count is not None:
                src.Log.print_with_color(f"Label distribution of client: {self.label_count}", "yellow")

            # Load training dataset
            if self.layer_id == 1 and data_name and not self.train_set and not self.label_to_indices:
                if data_name == "MNIST":
                    transform_train = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ])
                    self.train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                                                transform=transform_train)

                elif data_name == "CIFAR10":
                    transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ])
                    self.train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                                  transform=transform_train)
                elif data_name == "SPEECHCOMMANDS":
                    from src.dataset.SPEECHCOMMANDS import SpeechCommandsDataset
                    self.train_set = SpeechCommandsDataset(root='./data', subset='training')
                elif data_name == "AGNEWS":
                    from datasets import load_dataset
                    from transformers import BertTokenizer
                    from src.dataset.AGNEWS import AGNEWS_DATASET
                    
                    dataset = load_dataset('ag_news', download_mode='reuse_dataset_if_exists', cache_dir='./hf_cache')
                    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
                    
                    train_data = dataset['train']
                    texts = train_data['text']
                    labels = train_data['label']
                    
                    self.train_set = AGNEWS_DATASET(texts, labels, tokenizer, max_length=128)
                else:
                    self.train_set = None
                    raise ValueError(f"Data name '{data_name}' is not valid.")

                self.label_to_indices = defaultdict(list)
                if hasattr(self.train_set, 'labels'):
                    for idx, label in enumerate(self.train_set.labels):
                        self.label_to_indices[int(label)].append(idx)
                else:
                    for idx, (_, label) in tqdm(enumerate(self.train_set)):
                        self.label_to_indices[int(label)].append(idx)

            # Load model
            if self.model is None:
                klass = globals().get(f'{model_name}_{data_name}')
                if klass is None:
                    # try alternative names or mappings if needed
                    if model_name.upper() == 'BERT' and data_name == 'AGNEWS':
                        klass = Bert_AGNEWS
                    elif model_name == 'KWT':
                        klass = KWT_SPEECHCOMMANDS
                
                if klass is None:
                    raise ValueError(f"Model class for {model_name} and {data_name} not found.")

                if cut_layers[1] == -1:
                    self.model = klass(start_layer=cut_layers[0])
                else:
                    if klass == Bert_AGNEWS:
                         self.model = klass(layer_id=1, n_block=cut_layers[1]) if self.layer_id == 1 else klass(layer_id=2, n_block=12 - cut_layers[0])
                    else:
                         self.model = klass(start_layer=cut_layers[0], end_layer=cut_layers[1])

                self.model.to(self.device)

            batch_size = self.response["batch_size"]
            lr = self.response["lr"]
            momentum = self.response["momentum"]
            sda_size = self.response.get("sda_size", 1)
            layer2_devices = self.response.get("layer2_devices", [])

            # Read parameters and load to model
            if state_dict:
                self.model.load_state_dict(state_dict)

            # Apply LoRA for BERT model
            if model_name.upper() == 'BERT':
                if self.peft_config is None:
                    self.peft_config = LoraConfig(
                        task_type="SEQ_CLS",
                        r=8, lora_alpha=16, lora_dropout=0.1,
                        bias="none",
                        target_modules=["query", "key", "value", "dense"]
                    )
                self.model = get_peft_model(self.model, self.peft_config)
                # Note: layer15 might be a placeholder or refer to specific layers in user's model
                if self.layer_id == 2:
                    if hasattr(self.model, 'layer15'):
                        for param in self.model.layer15.parameters():
                            param.requires_grad = True

            self.model.to(self.device)

            # Start training
            if self.layer_id == 1:
                selected_indices = []
                for label, count in enumerate(self.label_count):
                    available = len(self.label_to_indices[label])
                    actual_count = min(count, available)
                    if actual_count > 0:
                        selected_indices.extend(random.sample(self.label_to_indices[label], actual_count))

                subset = torch.utils.data.Subset(self.train_set, selected_indices)
                train_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)

                result, size = self.train_func(self.model, lr, momentum, train_loader, local_round=local_round, layer2_devices=layer2_devices, model_name=model_name)

            else:
                result, size = self.train_func(self.model, lr, momentum, None, local_round=local_round, sda_size=sda_size, model_name=model_name)

            # Merge LoRA weights back for BERT
            if model_name.upper() == 'BERT':
                self.model = self.model.merge_and_unload()

            # Stop training, then send parameters to server
            model_state_dict = copy.deepcopy(self.model.state_dict())
            if self.device != "cpu":
                for key in model_state_dict:
                    model_state_dict[key] = model_state_dict[key].to('cpu')

            data = {"action": "UPDATE", "client_id": self.client_id, "layer_id": self.layer_id,
                    "result": result, "size": size,
                    "message": "Sent parameters to Server", "parameters": model_state_dict}
            src.Log.print_with_color("[>>>] Client sent parameters to server", "red")
            self.send_to_server(data)
            return True
        elif action == "STOP":
            return False
        return True


    def send_to_server(self, message):
        self.channel.queue_declare('rpc_queue', durable=False)
        self.channel.basic_publish(exchange='',
                                   routing_key='rpc_queue',
                                   body=pickle.dumps(message))
