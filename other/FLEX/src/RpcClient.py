import time
import pickle
import copy

import src.Log
from src.model.Bert_AGNEWS import Bert_AGNEWS
from src.model.VGG16_CIFAR10 import VGG16_CIFAR10
from src.model.KWT_SPEECHCOMMANDS import KWT_SPEECHCOMMANDS
from src.train.VGG16 import Train_VGG16
from src.train.Bert import Train_Bert
from src.train.KWT import Train_KWT
from src.dataset.dataloader import data_loader

from peft import LoraConfig, TaskType, get_peft_model

class RpcClient:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.model_train = None
        self.train_loader = None
        self.device = device

        self.response = None
        self.model = None
        self.cluster = None
        self.label_count = None
        self.peft_config = None

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
            model_name = self.response["model_name"]
            cut_layer = self.response['cut_layer']
            self.label_count = self.response['label_count']
            data_name = self.response["data_name"]
            self.cluster = self.response['cluster']

            if model_name == 'VGG16':
                self.model_train = Train_VGG16(self.client_id, self.layer_id, self.channel, self.device)
            elif model_name == 'Bert':
                self.model_train = Train_Bert(self.client_id, self.layer_id, self.channel, self.device)
                self.peft_config = LoraConfig(
                    task_type="SEQ_CLS",
                    r=8, lora_alpha=16, lora_dropout=0.1,
                    bias="none",
                    target_modules=["query", "key", "value", "dense"]
                )
            else:
                self.model_train = Train_KWT(self.client_id, self.layer_id, self.channel, self.device)

            if self.label_count is not None:
                src.Log.print_with_color(f"Label distribution of client: {self.label_count}", "yellow")

            # Load model
            if self.model is None:
                if model_name != 'Bert':
                    if model_name == 'VGG16':
                        klass = VGG16_CIFAR10
                    else:
                        klass = KWT_SPEECHCOMMANDS

                    if self.layer_id == 1:
                        self.model = klass(end_layer=cut_layer)
                    else:
                        self.model = klass(start_layer=cut_layer)
                else:
                    klass = Bert_AGNEWS
                    if self.layer_id == 1:
                        self.model = klass(layer_id=1, n_block=cut_layer)
                    else:
                        self.model = klass(layer_id=2, n_block=12 - cut_layer)

            learning = self.response["learning"]
            batch_size = learning["batch-size"]

            if state_dict is not None:
                self.model.load_state_dict(state_dict)

            if model_name == 'Bert':
                self.model = get_peft_model(self.model, self.peft_config)
                if self.layer_id == 2:
                    for param in self.model.classifier.parameters():
                        param.requires_grad = True

            self.model.to(self.device)

            if self.layer_id == 1:
                if self.train_loader is None:
                    self.train_loader = data_loader(data_name, batch_size, self.label_count, train=True)

                result, size, send = self.model_train.train_on_first_layer(self.model, learning ,self.train_loader, self.cluster)

            else:
                result, size, send = self.model_train.train_on_last_layer(self.model, learning, self.cluster)

            if model_name == 'Bert':
                self.model = self.model.merge_and_unload()
            if send:
                model_state_dict = copy.deepcopy(self.model.state_dict())
                if self.device != "cpu":
                    for key in model_state_dict:
                        model_state_dict[key] = model_state_dict[key].to('cpu')
            else:
                model_state_dict = None
            data = {"action": "UPDATE", "client_id": self.client_id, "layer_id": self.layer_id,
                    "result": result, "size": size, "cluster": self.cluster,
                    "message": "Sent parameters to Server", "parameters": model_state_dict}
            src.Log.print_with_color("[>>>] Client sent parameters to server", "red")
            self.send_to_server(data)
            return True
        elif action == "STOP":
            return False

    def send_to_server(self, message):
        self.response = None

        self.channel.queue_declare('rpc_queue', durable=False)
        self.channel.basic_publish(exchange='',
                                   routing_key='rpc_queue',
                                   body=pickle.dumps(message))

        return self.response
