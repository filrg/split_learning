import time
import pickle
import copy

import src.Log

from src.model import *
from src.dataset.dataloader import data_loader

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
        self.train_loader = None
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
        src.Log.print_with_color(f"[<<<] Client received: {self.response['message']}", "blue")
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

            if self.model is None:

                klass = globals()[f'{model_name}_{data_name}']

                if cut_layers[1] == -1:
                    self.model = klass(start_layer=cut_layers[0])
                else:
                    self.model = klass(start_layer=cut_layers[0], end_layer=cut_layers[1])

                self.model.to(self.device)

            batch_size = self.response["batch_size"]
            lr = self.response["lr"]
            momentum = self.response["momentum"]
            sda_size = self.response.get("sda_size", 1)
            layer2_devices = self.response.get("layer2_devices", [])

            if state_dict:
                self.model.load_state_dict(state_dict)

            if model_name == 'BERT':
                if self.peft_config is None:
                    self.peft_config = LoraConfig(
                        task_type="SEQ_CLS",
                        r=8, lora_alpha=16, lora_dropout=0.1,
                        bias="none",
                        target_modules=["query", "key", "value", "dense"]
                    )
                self.model = get_peft_model(self.model, self.peft_config)
                if self.layer_id == 2:
                    for param in self.model.layer15.parameters():
                        param.requires_grad = True

            self.model.to(self.device)

            if self.layer_id == 1:
                if self.train_loader is None:
                    self.train_loader = data_loader(data_name, batch_size, self.label_count, train=True)

                result, size = self.train_func(self.model, lr, momentum, self.train_loader, local_round=local_round, layer2_devices=layer2_devices, model_name=model_name)

            else:
                result, size = self.train_func(self.model, lr, momentum, None, local_round=local_round, sda_size=sda_size, model_name=model_name)

            if model_name == 'BERT':
                self.model = self.model.merge_and_unload()

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

    def send_to_server(self, message):
        self.response = None

        self.channel.queue_declare('rpc_queue', durable=False)
        self.channel.basic_publish(exchange='',
                                   routing_key='rpc_queue',
                                   body=pickle.dumps(message))

        return self.response
