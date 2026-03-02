import time
import pickle

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

    def send_intermediate_output(self, output, labels, trace):

        forward_queue_name = f'intermediate_queue_{self.layer_id}'

        self.channel.queue_declare(forward_queue_name, durable=False)

        if trace:
            trace.append(self.client_id)
            message = pickle.dumps(
                {"data": output.detach().cpu().numpy(), "label": labels.cpu().numpy(),
                 "trace": trace}
            )
        else:
            message = pickle.dumps(
                {"data": output.detach().cpu().numpy(), "label": labels.cpu().numpy(),
                 "trace": [self.client_id]}
            )

        self.channel.basic_publish(
            exchange='',
            routing_key=forward_queue_name,
            body=message
        )

    def send_gradient(self, gradient, trace):
        to_client_id = trace[-1]
        trace.pop(-1)
        backward_queue_name = f'gradient_queue_{self.layer_id - 1}_{to_client_id}'
        self.channel.queue_declare(queue=backward_queue_name, durable=False)

        message = pickle.dumps(
            {"data": gradient.detach().cpu().numpy(), "trace": trace})

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

    def train_on_first_layer(self, model, lr, momentum, train_loader=None, local_round=3):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        backward_queue_name = f'gradient_queue_{self.layer_id}_{self.client_id}'
        self.channel.queue_declare(queue=backward_queue_name, durable=False)
        self.channel.basic_qos(prefetch_count=1)

        model.to(self.device)
        start_time = time.time()

        for i in range(local_round):
            data_iter = iter(train_loader)
            src.Log.print_with_color(f'Forward epoch {i}', 'green')

            with tqdm(total=len(train_loader), desc="Processing", unit="step") as pbar:
                while True:
                    try:
                        training_data, labels = next(data_iter)
                        training_data = training_data.to(self.device)
                        intermediate_output = model(training_data)
                        intermediate_output = intermediate_output.detach().requires_grad_(True)

                        self.data_count += 1

                        self.send_intermediate_output(intermediate_output, labels, trace=None)
                        while True:
                            model.train()
                            optimizer.zero_grad()
                            method_frame, header_frame, body = self.channel.basic_get(queue=backward_queue_name, auto_ack=True)
                            if method_frame and body:
                                received_data = pickle.loads(body)
                                gradient_numpy = received_data["data"]
                                gradient = torch.tensor(gradient_numpy).to(self.device)

                                output = model(training_data)
                                output.backward(gradient=gradient)
                                optimizer.step()
                                break
                            else:
                                time.sleep(0.5)
                                continue

                        pbar.update(1)

                    except StopIteration:
                        break

        # Finish epoch training, send notify to server
        end_time = time.time()
        elapsed_time = end_time - start_time
        processing_speed = self.data_count / elapsed_time if elapsed_time > 0 else 0
        
        notify_data = {"action": "NOTIFY", "client_id": self.client_id, "layer_id": self.layer_id,
                       "message": "Finish training!", "processing_speed": processing_speed}

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

    def train_on_last_layer(self, model, lr, momentum):

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        result = True

        criterion = nn.CrossEntropyLoss()

        forward_queue_name = f'intermediate_queue_{self.layer_id - 1}'

        self.channel.queue_declare(queue=forward_queue_name, durable=False)
        self.channel.basic_qos(prefetch_count=1)
        print('Waiting for intermediate output. To exit press CTRL+C')
        model.to(self.device)
        infor_data = []
        tensor_data = torch.tensor([])
        tensor_label = torch.tensor([])
        count = 0

        while True:

            method_frame, header_frame, body = self.channel.basic_get(queue=forward_queue_name, auto_ack=True)
            if method_frame and body:
                received_data = pickle.loads(body)
                intermediate_output_numpy = received_data["data"]
                labels_numpy = received_data["label"]
                trace = received_data["trace"]

                labels = torch.tensor(labels_numpy)
                intermediate_output = torch.tensor(intermediate_output_numpy, requires_grad=True)

                infor_data.append((trace, intermediate_output.size(0)))
                tensor_data = torch.cat((tensor_data, intermediate_output), dim=0)
                tensor_label = torch.cat((tensor_label, labels), dim=0)

                self.data_count += 1
                count += 1
            else:
                if count == 3:
                    model.train()
                    optimizer.zero_grad()
                    tensor_data = tensor_data.to(self.device)
                    tensor_data.retain_grad()
                    tensor_label = tensor_label.to(self.device)

                    output = model(tensor_data)
                    loss = criterion(output, tensor_label.long())
                    print(f"Loss: {loss.item()}")

                    if torch.isnan(loss).any():
                        src.Log.print_with_color("NaN detected in loss", "yellow")
                        result = False

                    loss.backward()

                    optimizer.step()
                    gradient = tensor_data.grad

                    for (trace, size) in infor_data:
                        grad, new_gradient = gradient.split([size, gradient.size(0) - size], dim=0)
                        gradient = new_gradient

                        self.send_gradient(grad, trace)

                    infor_data = []
                    tensor_data = torch.tensor([])
                    tensor_label = torch.tensor([])
                    count = 0

                else:
                    broadcast_queue_name = f'reply_{self.client_id}'
                    method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
                    if body:
                        received_data = pickle.loads(body)
                        src.Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                        if received_data["action"] == "PAUSE":
                            return result

    def train_on_device(self, model, lr, momentum, train_loader=None, local_round=None):
        self.data_count = 0
        if self.layer_id == 1:
            result = self.train_on_first_layer(model, lr, momentum, train_loader,local_round)
        else:
            result = self.train_on_last_layer(model, lr, momentum)

        return result, self.data_count