import time
import uuid
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn

import src.Log
class Train_BERT:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.device = device
        self.data_count = 0
        self.size = None

    def send_intermediate_output(self, data_id, output, labels, trace, cluster=None):

        forward_queue_name = f'intermediate_queue_{self.layer_id}_{cluster}'
        self.channel.queue_declare(forward_queue_name, durable=False)

        if trace:
            trace.append(self.client_id)
            message = pickle.dumps(
                {"data_id": data_id, "data": output.detach().cpu().numpy(), "label": labels.cpu(), "trace": trace}
            )
        else:
            message = pickle.dumps(
                {"data_id": data_id, "data": output.detach().cpu().numpy(), "label": labels.cpu(), "trace": [self.client_id]}
            )

        # if self.size is None:
        #     self.size = len(message)
        #     print(f'Length message: {self.size} (bytes).')

        self.channel.basic_publish(
            exchange='',
            routing_key=forward_queue_name,
            body=message
        )

    def send_gradient(self, data_id, gradient, trace):
        to_client_id = trace[-1]
        trace.pop(-1)
        backward_queue_name = f'gradient_queue_{self.layer_id - 1}_{to_client_id}'
        self.channel.queue_declare(queue=backward_queue_name, durable=False)

        message = pickle.dumps(
            {"data_id": data_id, "data": gradient.detach().cpu().numpy(), "trace": trace})

        if self.size is None:
            self.size = len(message)
            print(f'Length message: {self.size} (bytes).')
        self.channel.basic_publish(
            exchange='',
            routing_key=backward_queue_name,
            body=message
        )

    def send_to_server(self, message):
        self.channel.queue_declare('rpc_queue', durable=False)
        self.channel.basic_publish(exchange='',
                                   routing_key='rpc_queue',
                                   body=pickle.dumps(message))

    def train_on_first_layer(self, model, learning, train_loader=None, cluster=0):
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning['learning-rate'], weight_decay=learning['weight-decay'])

        backward_queue_name = f'gradient_queue_{self.layer_id}_{self.client_id}'
        self.channel.queue_declare(queue=backward_queue_name, durable=False)
        self.channel.basic_qos(prefetch_count=1)
        model = model.to(self.device)

        data_iter = iter(train_loader)
        num_forward = 0
        num_backward = 0
        end_data = False
        data_store = {}

        with tqdm(total=len(train_loader), desc="Processing", unit="step") as pbar:
            while True:
                # Training model
                model.train()
                optimizer.zero_grad()

                # Process gradient
                method_frame, header_frame, body = self.channel.basic_get(queue=backward_queue_name, auto_ack=True)
                if method_frame and body:
                    num_backward += 1
                    received_data = pickle.loads(body)
                    gradient_numpy = received_data["data"]
                    gradient = torch.tensor(gradient_numpy).to(self.device)
                    data_id = received_data["data_id"]

                    data_input = data_store.pop(data_id)
                    output = model(input_ids=data_input)
                    output.backward(gradient=gradient)
                    optimizer.step()
                else:
                    # speed control
                    if len(data_store) >= learning['control-count']:
                        continue
                    else:
                        try:
                            batch = next(data_iter)
                            input_ids = batch['input_ids'].to(self.device)
                            labels = batch['labels'].to(self.device)
                            data_id = uuid.uuid4()
                            data_store[data_id] = input_ids

                            intermediate_output = model(input_ids=input_ids)
                            intermediate_output = intermediate_output.detach().requires_grad_(True)

                            num_forward += 1
                            self.data_count += 1

                            pbar.update(1)
                            self.send_intermediate_output(data_id, intermediate_output, labels, trace=None, cluster=cluster)

                        except StopIteration:
                            end_data = True

                if end_data and (num_forward == num_backward):
                    break

        notify_data = {"action": "NOTIFY", "client_id": self.client_id, "layer_id": self.layer_id,
                       "message": "Finish training!", "cluster": cluster}

        src.Log.print_with_color("[>>>] Finish training!", "red")
        self.send_to_server(notify_data)

        broadcast_queue_name = f'reply_{self.client_id}'
        while True:
            method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
            if body:
                received_data = pickle.loads(body)
                src.Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                if received_data["action"] == "PAUSE":
                    return True, self.data_count
            time.sleep(0.5)

    def train_on_last_layer(self, model, learning, cluster=0):
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning['learning-rate'], weight_decay=learning['weight-decay'])
        criterion = nn.CrossEntropyLoss()
        result = True

        forward_queue_name = f'intermediate_queue_{self.layer_id - 1}_{cluster}'
        self.channel.queue_declare(queue=forward_queue_name, durable=False)
        self.channel.basic_qos(prefetch_count=1)
        print('Waiting for intermediate output. To exit press CTRL+C')
        model.to(self.device)
        model.train()
        while True:
            method_frame, header_frame, body = self.channel.basic_get(queue=forward_queue_name, auto_ack=True)
            if method_frame and body:

                optimizer.zero_grad()
                received_data = pickle.loads(body)
                intermediate_output_numpy = received_data["data"]
                trace = received_data["trace"]
                data_id = received_data["data_id"]
                labels = received_data["label"].to(self.device)
                intermediate_output = torch.tensor(intermediate_output_numpy, requires_grad=True).float().to(self.device)

                output = model(input_ids=intermediate_output)

                loss = criterion(output, labels)

                if torch.isnan(loss).any():
                    src.Log.print_with_color("NaN detected in loss", "yellow")
                    result = False

                print(f"Loss: {loss.item()}")
                intermediate_output.retain_grad()
                loss.backward()

                optimizer.step()
                self.data_count += 1

                gradient = intermediate_output.grad
                self.send_gradient(data_id, gradient, trace)

            else:
                broadcast_queue_name = f'reply_{self.client_id}'
                method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
                if body:
                    received_data = pickle.loads(body)
                    src.Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                    if received_data["action"] == "PAUSE":
                        return result, self.data_count
