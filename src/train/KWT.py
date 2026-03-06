import time
import uuid
import pickle
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn

import src.Log


class Train_KWT:
    """Training class for Keyword Transformer (KWT) model - same logic as Train_ViT"""
    
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.device = device
        self.data_count = 0

    def send_intermediate_output(self, data_id, output, labels, trace, test=False, cluster=None):
        forward_queue_name = f'intermediate_queue_{self.layer_id}_{cluster}'
        self.channel.queue_declare(forward_queue_name, durable=False)

        if trace:
            trace.append(self.client_id)
            message = pickle.dumps(
                {"data_id": data_id, "data": output.detach().cpu().numpy(), "label": labels, "trace": trace,
                 "test": test}
            )
        else:
            message = pickle.dumps(
                {"data_id": data_id, "data": output.detach().cpu().numpy(), "label": labels, "trace": [self.client_id],
                 "test": test}
            )

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
            {"data_id": data_id, "data": gradient.detach().cpu().numpy(), "trace": trace, "test": False})

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

    def train_on_first_layer(self, model, lr, momentum, clip_grad_norm, control_count=5,
                             train_loader=None, cluster=None, config_time=None):
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        mode_limited_time = config_time["enable"]
        limited_time = config_time["time"]
        if mode_limited_time:
            epoch = 100
        else:
            epoch = config_time["epoch"]

        backward_queue_name = f'gradient_queue_{self.layer_id}_{self.client_id}'
        self.channel.queue_declare(queue=backward_queue_name, durable=False)
        self.channel.basic_qos(prefetch_count=1)
        start = time.time()
        model.to(self.device)

        for i in range(epoch):
            data_iter = iter(train_loader)
            num_forward = 0
            num_backward = 0
            end_data = False
            end_training = False
            data_store = {}

            with tqdm(total=len(train_loader), desc="Processing", unit="step") as pbar:
                while True:
                    model.train()
                    optimizer.zero_grad()
                    
                    method_frame, header_frame, body = self.channel.basic_get(queue=backward_queue_name, auto_ack=True)
                    if method_frame and body:
                        num_backward += 1
                        received_data = pickle.loads(body)
                        gradient_numpy = received_data["data"]
                        gradient = torch.tensor(gradient_numpy).to(self.device)
                        data_id = received_data["data_id"]

                        data_input = data_store.pop(data_id)
                        output = model(data_input)
                        output.backward(gradient=gradient)
                        optimizer.step()
                    else:
                        if len(data_store) >= control_count:
                            continue
                        
                        if ((time.time() - start) > limited_time) and mode_limited_time is True:
                            if i > 0:
                                end_data = True
                                end_training = True
                            else:
                                if num_forward == num_backward:
                                    end_training = True
                                    break
                        else:
                            try:
                                training_data, labels = next(data_iter)
                                training_data = training_data.to(self.device)
                                data_id = uuid.uuid4()
                                data_store[data_id] = training_data
                                intermediate_output = model(training_data)
                                intermediate_output = intermediate_output.detach().requires_grad_(True)

                                num_forward += 1
                                self.data_count += 1
                                pbar.update(1)

                                self.send_intermediate_output(data_id, intermediate_output, labels, trace=None, test=False, cluster=cluster)

                            except StopIteration:
                                end_data = True

                    if end_data and (num_forward == num_backward):
                        if mode_limited_time is False:
                            break

                if end_training is True:
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

    def train_on_last_layer(self, model, lr, momentum, clip_grad_norm, cluster):
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        result = True

        criterion = nn.CrossEntropyLoss()
        forward_queue_name = f'intermediate_queue_{self.layer_id - 1}_{cluster}'

        self.channel.queue_declare(queue=forward_queue_name, durable=False)
        self.channel.basic_qos(prefetch_count=1)
        print('Waiting for intermediate output. To exit press CTRL+C')
        model.to(self.device)
        
        while True:
            model.train()
            optimizer.zero_grad()
            
            method_frame, header_frame, body = self.channel.basic_get(queue=forward_queue_name, auto_ack=True)
            if method_frame and body:
                received_data = pickle.loads(body)
                intermediate_output_numpy = received_data["data"]
                trace = received_data["trace"]
                data_id = received_data["data_id"]
                labels = received_data["label"].to(self.device)

                intermediate_output = torch.tensor(intermediate_output_numpy, requires_grad=True).to(self.device)

                output = model(intermediate_output)

                loss = criterion(output, labels)
                print(f"Loss: {loss.item()}")
                if torch.isnan(loss).any():
                    src.Log.print_with_color("NaN detected in loss", "yellow")
                    result = False

                intermediate_output.retain_grad()
                loss.backward()
                if clip_grad_norm and clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
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

    def train_on_middle_layer(self, model, lr, momentum, clip_grad_norm, control_count=5, cluster=None):
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        forward_queue_name = f'intermediate_queue_{self.layer_id - 1}'
        backward_queue_name = f'gradient_queue_{self.layer_id}_{self.client_id}'
        self.channel.queue_declare(queue=forward_queue_name, durable=False)
        self.channel.queue_declare(queue=backward_queue_name, durable=False)
        self.channel.basic_qos(prefetch_count=1)
        data_store = {}
        print('Waiting for intermediate output. To exit press CTRL+C')
        model.to(self.device)
        
        while True:
            model.train()
            optimizer.zero_grad()
            
            method_frame, header_frame, body = self.channel.basic_get(queue=backward_queue_name, auto_ack=True)
            if method_frame and body:
                received_data = pickle.loads(body)
                gradient_numpy = received_data["data"]
                gradient = torch.tensor(gradient_numpy).to(self.device)
                trace = received_data["trace"]
                data_id = received_data["data_id"]

                data_input = data_store.pop(data_id)
                output = model(data_input)
                data_input.retain_grad()
                output.backward(gradient=gradient, retain_graph=True)
                optimizer.step()

                gradient = data_input.grad
                self.send_gradient(data_id, gradient, trace)
            else:
                method_frame, header_frame, body = self.channel.basic_get(queue=forward_queue_name, auto_ack=True)
                if method_frame and body:
                    received_data = pickle.loads(body)
                    intermediate_output_numpy = received_data["data"]
                    trace = received_data["trace"]
                    data_id = received_data["data_id"]
                    test = received_data["test"]
                    labels = received_data["label"].to(self.device)

                    intermediate_output = torch.tensor(intermediate_output_numpy, requires_grad=True).to(self.device)
                    data_store[data_id] = intermediate_output

                    output = model(intermediate_output)
                    output = output.detach().requires_grad_(True)

                    self.data_count += 1
                    self.send_intermediate_output(data_id, output, labels, trace, test, cluster=cluster)
                    
                    if len(data_store) > control_count:
                        continue
                        
            if method_frame is None:
                broadcast_queue_name = f'reply_{self.client_id}'
                method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
                if body:
                    received_data = pickle.loads(body)
                    src.Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                    if received_data["action"] == "PAUSE":
                        return True, self.data_count

    def alone_training(self, model, lr, momentum, clip_grad_norm, train_loader=None, cluster=None):
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        print('Waiting for training. To exit press CTRL+C')
        
        for training_data, labels in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()
            training_data = training_data.to(self.device)
            labels = labels.to(self.device)
            output = model(training_data)

            loss = criterion(output, labels)
            if torch.isnan(loss).any():
                src.Log.print_with_color("NaN detected in loss", "yellow")
            loss.backward()
            if clip_grad_norm and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            self.data_count += 1

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
