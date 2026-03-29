import time
import pickle
import uuid

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

import src.Log
from tqdm import tqdm

class Scheduler:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.device = device
        self.data_count = 0

    def send_intermediate_output(self, output, labels, trace, data_id=None, target_device_id=None):

        if target_device_id is not None:
            forward_queue_name = f'intermediate_queue_{target_device_id}'
        else:
            forward_queue_name = f'intermediate_queue_{self.layer_id}'

        self.channel.queue_declare(forward_queue_name, durable=False)

        if trace:
            trace.append(self.client_id)
            message = pickle.dumps(
                {"data": output.detach().cpu().numpy(), "label": labels.cpu().numpy(),
                 "trace": trace, "data_id": data_id}
            )
        else:
            message = pickle.dumps(
                {"data": output.detach().cpu().numpy(), "label": labels.cpu().numpy(),
                 "trace": [self.client_id], "data_id": data_id}
            )

        self.channel.basic_publish(
            exchange='',
            routing_key=forward_queue_name,
            body=message
        )

    def send_gradient(self, gradient, trace, data_id=None):
        to_client_id = trace[-1]
        trace.pop(-1)
        backward_queue_name = f'gradient_queue_{self.layer_id - 1}_{to_client_id}'
        self.channel.queue_declare(queue=backward_queue_name, durable=False)

        message = pickle.dumps(
            {"data": gradient.detach().cpu().numpy(), "trace": trace, "data_id": data_id})

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

    def train_on_first_layer(self, model, lr, momentum, train_loader=None, local_round=3, layer2_devices=None, model_name=None):
        if model_name == 'BERT':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        backward_queue_name = f'gradient_queue_{self.layer_id}_{self.client_id}'
        self.channel.queue_declare(queue=backward_queue_name, durable=False)
        self.channel.basic_qos(prefetch_count=1)

        model.to(self.device)

        batch_counter = 0

        for i in range(local_round):
            src.Log.print_with_color(f'Epoch {i}', 'green')

            with tqdm(total=len(train_loader), desc="Processing", unit="step") as pbar:
                for batch in train_loader:
                    if isinstance(batch, dict) and 'input_ids' in batch:
                        training_data = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        kwargs = {'input_ids': training_data, 'attention_mask': attention_mask}
                    else:
                        training_data, labels = batch
                        training_data = training_data.to(self.device)
                        labels = labels.to(self.device)
                        kwargs = {}

                    data_id = str(uuid.uuid4())
                    with torch.no_grad():
                        if 'input_ids' in kwargs:
                            intermediate_output = model(**kwargs)
                        else:
                            intermediate_output = model(training_data, **kwargs)
                    intermediate_output = intermediate_output.detach().requires_grad_(True)

                    self.data_count += 1
                    pbar.update(1)

                    target_device_id = None
                    if layer2_devices:
                        target_device_id = layer2_devices[batch_counter % len(layer2_devices)]
                        batch_counter += 1

                    self.send_intermediate_output(intermediate_output, labels, trace=None, data_id=data_id, target_device_id=target_device_id)

                    while True:
                        method_frame, header_frame, body = self.channel.basic_get(
                            queue=backward_queue_name, auto_ack=True)
                        if method_frame and body:
                            received_data = pickle.loads(body)
                            gradient = torch.tensor(received_data["data"]).to(self.device)

                            model.train()
                            optimizer.zero_grad()
                            if 'input_ids' in kwargs:
                                output = model(**kwargs)
                            else:
                                output = model(training_data, **kwargs)
                            output.backward(gradient=gradient)
                            optimizer.step()
                            break
                        time.sleep(0.01)

        # Finish training, notify server
        notify_data = {"action": "NOTIFY", "client_id": self.client_id, "layer_id": self.layer_id,
                       "message": "Finish training!"}
        src.Log.print_with_color("[>>>] Finish training!", "red")
        self.send_to_server(notify_data)

        # Wait for PAUSE
        broadcast_queue_name = f'reply_{self.client_id}'
        while True:
            method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
            if body:
                received_data = pickle.loads(body)
                src.Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                if received_data["action"] == "PAUSE":
                    return True
            time.sleep(0.5)

    def _process_sda_batch(self, model, optimizer, criterion, collected, model_name=None):
        batch_sizes = [item["data"].shape[0] for item in collected]
        traces = [item["trace"] for item in collected]
        data_ids = [item["data_id"] for item in collected]

        all_data = np.concatenate([item["data"] for item in collected], axis=0)
        all_labels = np.concatenate([item["label"] for item in collected], axis=0)

        concat_intermediate = torch.tensor(all_data, requires_grad=True).to(self.device)
        concat_labels = torch.tensor(all_labels).to(self.device)

        model.train()
        optimizer.zero_grad()
        concat_intermediate.retain_grad()

        if model_name == 'BERT':
            output = model(input_ids=concat_intermediate)
        else:
            output = model(concat_intermediate)
        loss = criterion(output, concat_labels.long())
        print(f"Loss (SDA, {len(collected)} clients, {sum(batch_sizes)} samples): {loss.item():.4f}")

        result = True
        if torch.isnan(loss).any():
            src.Log.print_with_color("NaN detected in loss", "yellow")
            result = False

        loss.backward()
        optimizer.step()

        self.data_count += sum(batch_sizes)

        # Split gradient back to each client
        concat_grad = concat_intermediate.grad
        grad_splits = torch.split(concat_grad, batch_sizes, dim=0)

        for grad, trace, data_id in zip(grad_splits, traces, data_ids):
            self.send_gradient(grad, trace, data_id=data_id)

        return result

    def train_on_last_layer(self, model, lr, momentum, sda_size=1, model_name=None):
        if model_name == 'BERT':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        result = True
        criterion = nn.CrossEntropyLoss()

        forward_queue_name = f'intermediate_queue_{self.client_id}'
        self.channel.queue_declare(queue=forward_queue_name, durable=False)
        self.channel.basic_qos(prefetch_count=1)
        print(f'Waiting for intermediate output on queue {forward_queue_name} (SDA size={sda_size}). To exit press CTRL+C')
        model.to(self.device)

        sda_batch = {}  # {client_id: data} — exactly 1 batch per client

        while True:
            method_frame, header_frame, body = self.channel.basic_get(queue=forward_queue_name, auto_ack=True)
            if method_frame and body:
                received_data = pickle.loads(body)
                client_id = received_data["trace"][0]
                sda_batch[client_id] = received_data

                # When we have 1 batch from each client → SDA forward
                if len(sda_batch) >= sda_size:
                    batch_result = self._process_sda_batch(model, optimizer, criterion, list(sda_batch.values()), model_name=model_name)
                    if not batch_result:
                        result = False
                    sda_batch = {}
            else:
                # Check for PAUSE
                broadcast_queue_name = f'reply_{self.client_id}'
                method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
                if body:
                    received_data = pickle.loads(body)
                    src.Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                    if received_data["action"] == "PAUSE":
                        # Process remaining
                        if sda_batch:
                            batch_result = self._process_sda_batch(model, optimizer, criterion, list(sda_batch.values()), model_name=model_name)
                            if not batch_result:
                                result = False
                        return result

    def train_on_device(self, model, lr, momentum, train_loader=None, local_round=None, sda_size=1, layer2_devices=None, model_name=None):
        self.data_count = 0
        if self.layer_id == 1:
            result = self.train_on_first_layer(model, lr, momentum, train_loader, local_round, layer2_devices=layer2_devices, model_name=model_name)
        else:
            result = self.train_on_last_layer(model, lr, momentum, sda_size, model_name=model_name)

        return result, self.data_count