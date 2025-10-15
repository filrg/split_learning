import time
import uuid
import pickle
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn

import src.Log


class Scheduler:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.device = device
        self.data_count = 0

    def send_intermediate_output(self, data_id, label_count, output, labels, trace):

        forward_queue_name = f'intermediate_queue_{self.layer_id}'

        self.channel.queue_declare(forward_queue_name, durable=False)

        if trace:
            trace.append(self.client_id)
            message = pickle.dumps(
                {"data_id": data_id, "label_count": label_count, "data": output.detach().cpu().numpy(), "label": labels, "trace": trace}
            )
        else:
            message = pickle.dumps(
                {"data_id": data_id, "label_count": label_count, "data": output.detach().cpu().numpy(), "label": labels, "trace": [self.client_id]}
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
            {"data_id": data_id, "data": gradient.detach().cpu().numpy(), "trace": trace})

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

    def train_on_first_layer(self, model, label_count, lr, momentum, clip_grad_norm, control_count=5,
                             train_loader=None, config_time=None):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        mode_limited_time = config_time["enable"]
        limited_time = config_time["time"]

        backward_queue_name = f'gradient_queue_{self.layer_id}_{self.client_id}'
        self.channel.queue_declare(queue=backward_queue_name, durable=False)
        self.channel.basic_qos(prefetch_count=1)
        start = time.time()
        model.to(self.device)

        for i in range(100):
            data_iter = iter(train_loader)
            num_forward = 0
            num_backward = 0
            end_data = False
            end_training = False
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
                        output = model(data_input)
                        output.backward(gradient=gradient)
                        optimizer.step()
                    else:
                        # speed control
                        if len(data_store) > control_count:
                            continue
                        # Process forward message
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

                                # Send to next layers
                                num_forward += 1
                                self.data_count += 1
                                # tqdm bar
                                pbar.update(1)

                                self.send_intermediate_output(data_id, label_count, intermediate_output, labels, trace=None)

                            except StopIteration:
                                end_data = True

                    if end_data and (num_forward == num_backward):
                        if mode_limited_time is False:
                            end_training = True
                        break

                if end_training is True:
                    break

        notify_data = {"action": "NOTIFY", "client_id": self.client_id, "layer_id": self.layer_id,
                        "message": "Finish training!"}

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

    def train_on_last_layer(self, model, label_count, lr, momentum, clip_grad_norm):

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        result = True

        criterion = nn.CrossEntropyLoss()

        forward_queue_name = f'intermediate_queue_{self.layer_id - 1}'

        self.channel.queue_declare(queue=forward_queue_name, durable=False)
        self.channel.basic_qos(prefetch_count=1)
        print('Waiting for intermediate output. To exit press CTRL+C')
        model.to(self.device)
        while True:
            # Training model
            model.train()
            optimizer.zero_grad()
            # Process gradient
            method_frame, header_frame, body = self.channel.basic_get(queue=forward_queue_name, auto_ack=True)
            if method_frame and body:

                received_data = pickle.loads(body)
                intermediate_output_numpy = received_data["data"]
                trace = received_data["trace"]
                data_id = received_data["data_id"]
                labels = received_data["label"].to(self.device)
                label_count = received_data["label_count"]

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

    def train_on_device(self, model, label_count, lr, momentum, clip_grad_norm, num_layers, control_count, train_loader=None, config_time=None):
        self.data_count = 0
        if self.layer_id == 1:
            result = self.train_on_first_layer(model, label_count, lr, momentum, clip_grad_norm, control_count, train_loader, config_time)

        else:
            result = self.train_on_last_layer(model, label_count, lr, momentum, clip_grad_norm)

        return result, self.data_count
