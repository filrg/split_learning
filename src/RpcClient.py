import time
import pickle
import copy

import src.Log
from src.model.BERT_AGNEWS import BERT_AGNEWS
from src.model.KWT_SPEECHCOMMANDS import KWT_SPEECHCOMMANDS
from src.model.VGG16_CIFAR10 import VGG16_CIFAR10
from src.train.VGG16 import Train_VGG16
from src.train.BERT import Train_BERT
from src.train.KWT import Train_KWT
from src.dataset.dataloader import data_loader

from peft import LoraConfig, TaskType, get_peft_model

class RpcClient:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.model_train = None
        self.model_name = None
        self.train_loader = None
        self.device = device

        self.response = None
        self.model = None
        self.cluster = None
        self.learning = None
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

        if action == "START":
            state_dict = self.response["parameters"]
            self.model_name = self.response["model_name"]
            cut_layers = self.response['layers']
            label_count = self.response['label_count']
            refresh =  self.response['refresh']
            data_name = self.response["data_name"]
            self.learning = self.response["learning"]

            if self.model_name == 'VGG16':
                self.model_train = Train_VGG16(self.client_id, self.layer_id, self.channel, self.device)
            elif self.model_name == 'BERT':
                self.model_train = Train_BERT(self.client_id, self.layer_id, self.channel, self.device)
                self.peft_config = LoraConfig(
                    task_type="SEQ_CLS",
                    r=8, lora_alpha=16, lora_dropout=0.1,
                    bias="none",
                    target_modules=["query", "key", "value", "dense"]
                )
            elif self.model_name == 'KWT':
                self.model_train = Train_KWT(self.client_id, self.layer_id, self.channel, self.device)

            if self.label_count is None:
                self.label_count = label_count
            if self.response['cluster'] is not None:
                self.cluster = self.response['cluster']
            if self.label_count is not None:
                src.Log.print_with_color(f"Label distribution of client: {self.label_count}", "yellow")

            # Load model
            self.model = None
            if self.model_name == 'BERT':
                klass = BERT_AGNEWS
            elif self.model_name == 'KWT':
                klass = KWT_SPEECHCOMMANDS
            else:
                klass = VGG16_CIFAR10

            if cut_layers[1] != 0:
                if cut_layers[1] == -1:
                    self.model = klass(start_layer=cut_layers[0])
                else:
                    self.model = klass(start_layer=cut_layers[0], end_layer=cut_layers[1])
            else:
                self.model = klass()

            batch_size = self.learning["batch-size"]

            if state_dict:
                self.model.load_state_dict(state_dict)

            if self.model_name == 'BERT':
                self.model = get_peft_model(self.model, self.peft_config)
                if self.layer_id == 2:
                    for param in self.model.layer15.classifier.parameters():
                        param.requires_grad = True

            self.model.to(self.device)

            if self.layer_id == 1:
                if (self.train_loader is None) or refresh:
                    self.train_loader = data_loader(data_name, batch_size, self.label_count, train=True)

            return True

        elif action == 'SYN':
            # Start training
            if self.layer_id == 1:
                result, size = self.model_train.train_on_first_layer(self.model, self.learning, self.train_loader, self.cluster)
            else:
                result, size = self.model_train.train_on_last_layer(self.model, self.learning, self.cluster)

            # Stop training, then send parameters to server
            if self.model_name == 'BERT':
                self.model = self.model.merge_and_unload()

            model_state_dict = copy.deepcopy(self.model.state_dict())
            if self.device != "cpu":
                for key in model_state_dict:
                    model_state_dict[key] = model_state_dict[key].to('cpu')
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
