import time
import uuid
import pickle
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f

import src.Log


class Scheduler:
    def __init__(self, client_id, channel, device, args):
        self.client_id = client_id
        self.layer_id = args.layer_id
        self.channel = channel
        self.device = device
        self.data_count = 0

        self.event_time = args.event_time
        self.time_event_forward = []
        self.time_event_backward = []

    def balanced_softmax_loss(self, logits, labels, class_counts, epsilon=1e-6):
        class_counts = torch.tensor(class_counts, dtype=torch.int64).to(self.device)
        log_probs = f.log_softmax(logits, dim=1)
        class_probs = class_counts / (class_counts.sum() + epsilon)
        weights = 1.0 / (class_probs + epsilon)
        weights = weights / weights.sum()
        loss = (-weights[labels] * log_probs[range(labels.shape[0]), labels]).mean()
        return loss

    def send_intermediate_output(self, data_id, label_count, output, labels, trace, test=False, cluster=None, special=False):
        if special is True:
            forward_queue_name = f'intermediate_queue_{self.layer_id}'
        else:
            forward_queue_name = f'intermediate_queue_{self.layer_id}_{cluster}'
        self.channel.queue_declare(forward_queue_name, durable=False)

        if trace:
            trace.append(self.client_id)
            message = pickle.dumps(
                {"data_id": data_id, "label_count": label_count, "data": output.detach().cpu().numpy(), "label": labels, "trace": trace,
                 "test": test}
            )
        else:
            message = pickle.dumps(
                {"data_id": data_id, "label_count": label_count, "data": output.detach().cpu().numpy(), "label": labels, "trace": [self.client_id],
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

    def train_on_first_layer(self, model, global_model, label_count, lr, momentum, clip_grad_norm, compute_loss, control_count=5,
                             train_loader=None, cluster=None, special=False):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        data_iter = iter(train_loader)

        backward_queue_name = f'gradient_queue_{self.layer_id}_{self.client_id}'
        self.channel.queue_declare(queue=backward_queue_name, durable=False)
        self.channel.basic_qos(prefetch_count=10)
        num_forward = 0
        num_backward = 0
        end_data = False
        data_store = {}

        model.to(self.device)
        with tqdm(total=len(train_loader), desc="Processing", unit="step") as pbar:
            while True:
                # Training model
                model.train()
                optimizer.zero_grad()
                # Process gradient
                method_frame, header_frame, body = self.channel.basic_get(queue=backward_queue_name, auto_ack=True)
                if method_frame and body:
                    if self.event_time:
                        self.time_event_backward.append(time.time())
                    num_backward += 1
                    received_data = pickle.loads(body)
                    gradient_numpy = received_data["data"]
                    gradient = torch.tensor(gradient_numpy).to(self.device)
                    data_id = received_data["data_id"]

                    data_input = data_store.pop(data_id)
                    output = model(data_input)
                    output.backward(gradient=gradient)
                    optimizer.step()
                    if self.event_time:
                        self.time_event_backward.append(time.time())
                else:
                    # speed control
                    if len(data_store) > control_count:
                        continue
                    # Process forward message
                    try:
                        training_data, labels = next(data_iter)
                        if self.event_time:
                            self.time_event_forward.append(time.time())
                        training_data = training_data.to(self.device)
                        data_id = uuid.uuid4()
                        data_store[data_id] = training_data
                        intermediate_output = model(training_data)
                        intermediate_output = intermediate_output.detach().requires_grad_(True)
                        if self.event_time:
                            self.time_event_forward.append(time.time())

                        # Send to next layers
                        num_forward += 1
                        self.data_count += 1
                        # tqdm bar
                        pbar.update(1)

                        self.send_intermediate_output(data_id, label_count, intermediate_output, labels, trace=None, test=False, cluster=cluster, special=special)

                    except StopIteration:
                        end_data = True
                if end_data and (num_forward == num_backward):
                    break

            notify_data = {"action": "NOTIFY", "client_id": self.client_id, "layer_id": self.layer_id,
                           "message": "Finish training!", "cluster": cluster}

        # Finish epoch training, send notify to server
        src.Log.print_with_color("[>>>] Finish training!", "red")
        self.send_to_server(notify_data)

        broadcast_queue_name = f'reply_{self.client_id}'
        while True:  # Wait for broadcast
            method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
            if body:
                received_data = pickle.loads(body)
                src.Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                if received_data["action"] == "PAUSE":
                    return True
            time.sleep(0.5)

    def train_on_last_layer(self, model, global_model, label_count, lr, momentum, clip_grad_norm, compute_loss, cluster, special=False):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        result = True

        criterion = nn.CrossEntropyLoss()
        if special:
            forward_queue_name = f'intermediate_queue_{self.layer_id - 1}'
        else:
            forward_queue_name = f'intermediate_queue_{self.layer_id - 1}_{cluster}'
        self.channel.queue_declare(queue=forward_queue_name, durable=False)
        self.channel.basic_qos(prefetch_count=10)
        print('Waiting for intermediate output. To exit press CTRL+C')
        model.to(self.device)
        while True:
            # Training model
            model.train()
            optimizer.zero_grad()
            # Process gradient
            method_frame, header_frame, body = self.channel.basic_get(queue=forward_queue_name, auto_ack=True)
            if method_frame and body:
                if self.event_time:
                    self.time_event_forward.append(time.time())
                received_data = pickle.loads(body)
                intermediate_output_numpy = received_data["data"]
                trace = received_data["trace"]
                data_id = received_data["data_id"]
                labels = received_data["label"].to(self.device)
                label_count = received_data["label_count"]

                intermediate_output = torch.tensor(intermediate_output_numpy, requires_grad=True).to(self.device)

                output = model(intermediate_output)

                # choose loss mode
                if compute_loss["mode"] == 'FedProx':
                    loss = criterion(output, labels)
                    prox_term = 0.0
                    for param, global_param in zip(model.parameters(), global_model.parameters()):
                        prox_term += torch.norm(param - global_param, p=2)
                    loss += (compute_loss["FedProx"]["mu"] / 2) * prox_term
                elif compute_loss["mode"] == 'ReBaFL':
                    loss = self.balanced_softmax_loss(output, labels, label_count)
                    prox_term = sum(torch.norm(param - global_param, p=2) for param, global_param in
                                    zip(model.parameters(), global_model.parameters()))
                    loss += (compute_loss["ReBaFL"]["mu"] / 2) * prox_term
                    feature_aug_loss = compute_loss["ReBaFL"]["lambda_aug"] * torch.norm(output.mean(dim=0) - global_model(intermediate_output).mean(dim=0), p=2)
                    loss += feature_aug_loss
                else:
                    loss = criterion(output, labels)
                print(f"Loss: {loss.item()}")
                if torch.isnan(loss).any():
                    src.Log.print_with_color("NaN detected in loss", "yellow")
                    result = False

                if self.event_time:
                    self.time_event_forward.append(time.time())
                    self.time_event_backward.append(time.time())
                intermediate_output.retain_grad()
                loss.backward()
                if clip_grad_norm and clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
                self.data_count += 1

                gradient = intermediate_output.grad
                if self.event_time:
                    self.time_event_backward.append(time.time())
                self.send_gradient(data_id, gradient, trace)  # 1F1B
            # Check training process
            else:
                broadcast_queue_name = f'reply_{self.client_id}'
                method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
                if body:
                    received_data = pickle.loads(body)
                    src.Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                    if received_data["action"] == "PAUSE":
                        return result

    def train_on_middle_layer(self, model, global_model, label_count, lr, momentum, clip_grad_norm, compute_loss, control_count=5, cluster=None, special=False):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        forward_queue_name = f'intermediate_queue_{self.layer_id - 1}'
        backward_queue_name = f'gradient_queue_{self.layer_id}_{self.client_id}'
        self.channel.queue_declare(queue=forward_queue_name, durable=False)
        self.channel.queue_declare(queue=backward_queue_name, durable=False)
        self.channel.basic_qos(prefetch_count=10)
        data_store = {}
        print('Waiting for intermediate output. To exit press CTRL+C')
        model.to(self.device)
        while True:
            # Training model
            model.train()
            optimizer.zero_grad()
            # Process gradient
            method_frame, header_frame, body = self.channel.basic_get(queue=backward_queue_name, auto_ack=True)
            if method_frame and body:
                if self.event_time:
                    self.time_event_backward.append(time.time())
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
                if self.event_time:
                    self.time_event_backward.append(time.time())
                self.send_gradient(data_id, gradient, trace)
            else:
                method_frame, header_frame, body = self.channel.basic_get(queue=forward_queue_name, auto_ack=True)
                if method_frame and body:
                    if self.event_time:
                        self.time_event_forward.append(time.time())
                    received_data = pickle.loads(body)
                    intermediate_output_numpy = received_data["data"]
                    trace = received_data["trace"]
                    data_id = received_data["data_id"]
                    test = received_data["test"]
                    labels = received_data["label"].to(self.device)
                    label_count = received_data["label_count"]

                    intermediate_output = torch.tensor(intermediate_output_numpy, requires_grad=True).to(self.device)
                    data_store[data_id] = intermediate_output

                    output = model(intermediate_output)
                    output = output.detach().requires_grad_(True)

                    self.data_count += 1
                    if self.event_time:
                        self.time_event_forward.append(time.time())
                    self.send_intermediate_output(data_id, label_count, output, labels, trace, test, cluster=cluster, special=special)
                    # speed control
                    if len(data_store) > control_count:
                        continue
            # Check training process
            if method_frame is None:
                broadcast_queue_name = f'reply_{self.client_id}'
                method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
                if body:
                    received_data = pickle.loads(body)
                    src.Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                    if received_data["action"] == "PAUSE":
                        return True

    def alone_training(self, model, global_model, label_count, lr, momentum, clip_grad_norm, compute_loss, train_loader=None, cluster=None):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        criterion = nn.CrossEntropyLoss()
        print('Waiting for training. To exit press CTRL+C')
        for training_data, labels in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()
            training_data = training_data.to(self.device)
            labels = labels.to(self.device)
            output = model(training_data)

            if compute_loss["mode"] == 'FedProx':
                loss = criterion(output, labels)
                prox_term = 0.0
                for param, global_param in zip(model.parameters(), global_model.parameters()):
                    prox_term += torch.norm(param - global_param, p=2)
                loss += (compute_loss["FedProx"]["mu"] / 2) * prox_term
            elif compute_loss["mode"] == 'ReBaFL':
                loss = self.balanced_softmax_loss(output, labels, label_count)
                prox_term = sum(torch.norm(param - global_param, p=2) for param, global_param in
                                zip(model.parameters(), global_model.parameters()))
                loss += (compute_loss["ReBaFL"]["mu"] / 2) * prox_term
                feature_aug_loss = compute_loss["ReBaFL"]["lambda_aug"] * torch.norm(
                    output.mean(dim=0) - global_model(training_data).mean(dim=0), p=2)
                loss += feature_aug_loss
            else:
                loss = criterion(output, labels)
            if torch.isnan(loss).any():
                src.Log.print_with_color("NaN detected in loss", "yellow")
                result = False
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
        while True:  # Wait for broadcast
            method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
            if body:
                received_data = pickle.loads(body)
                src.Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                if received_data["action"] == "PAUSE":
                    return True
            time.sleep(0.5)

    def train_on_device(self, model, global_model, label_count, lr, momentum, clip_grad_norm, compute_loss, num_layers, control_count, train_loader=None, cluster=None, special=False, alone_train=False):
        self.data_count = 0
        if self.layer_id == 1:
            if alone_train is False:
                result = self.train_on_first_layer(model, global_model, label_count, lr, momentum, clip_grad_norm, compute_loss, control_count, train_loader, cluster, special)
            else:
                result = self.alone_training(model, global_model, label_count, lr, momentum, clip_grad_norm, compute_loss, train_loader=train_loader, cluster=cluster)
        elif self.layer_id == num_layers:
            result = self.train_on_last_layer(model, global_model, label_count, lr, momentum, clip_grad_norm, compute_loss, cluster=cluster, special=special)
        else:
            result = self.train_on_middle_layer(model, global_model, label_count, lr, momentum, clip_grad_norm, compute_loss, control_count, cluster=cluster, special=special)
        if self.event_time:
            src.Log.print_with_color(f"Forward training time events {self.time_event_forward}", "yellow")
            src.Log.print_with_color(f"Backward Training time events {self.time_event_backward}", "yellow")

        return result, self.data_count
