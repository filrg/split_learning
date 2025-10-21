import time
import pickle

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

    def send_intermediate_output(self, data_idx, output, labels, trace):

        forward_queue_name = f'intermediate_queue_{self.layer_id}'

        self.channel.queue_declare(forward_queue_name, durable=False)

        if trace:
            trace.append(self.client_id)
            message = pickle.dumps(
                {"data_idx": data_idx, "data": output.detach().cpu().numpy(), "label": labels.cpu().numpy(),
                 "trace": trace}
            )
        else:
            message = pickle.dumps(
                {"data_idx": data_idx, "data": output.detach().cpu().numpy(), "label": labels.cpu().numpy(),
                 "trace": [self.client_id]}
            )

        self.channel.basic_publish(
            exchange='',
            routing_key=forward_queue_name,
            body=message
        )

    def send_gradient(self, idx, gradient, trace):
        to_client_id = trace[-1]
        trace.pop(-1)
        backward_queue_name = f'gradient_queue_{self.layer_id - 1}_{to_client_id}'
        self.channel.queue_declare(queue=backward_queue_name, durable=False)

        message = pickle.dumps(
            {"idx": idx, "data": gradient.detach().cpu().numpy(), "trace": trace})

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

    def train_on_first_layer(self, model, lr, momentum, clip_grad_norm,
                             train_loader=None, config_time=None, local_round=5):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        size_store = 0
        backward_queue_name = f'gradient_queue_{self.layer_id}_{self.client_id}'
        self.channel.queue_declare(queue=backward_queue_name, durable=False)
        self.channel.basic_qos(prefetch_count=1)

        model.to(self.device)

        for i in range(local_round):
            data_iter = iter(train_loader)
            data_store = torch.tensor([])
            src.Log.print_with_color(f'Forward epoch {i}', 'green')
            while True:
                try:
                    training_data, labels = next(data_iter)
                    data_store = torch.cat((data_store, training_data), dim=0)
                    training_data = training_data.to(self.device)
                    intermediate_output = model(training_data)
                    intermediate_output = intermediate_output.detach().requires_grad_(True)

                    self.data_count += 1

                    self.send_intermediate_output(size_store, intermediate_output, labels, trace=None)
                    size_store += 1

                except StopIteration:
                    break

            list_grad = [torch.tensor([]) for _ in range(size_store)]
            grad_store = torch.tensor([])

            src.Log.print_with_color(f'Wait backward epoch {i}', 'green')
            count = 0
            while True:
                # Process gradient
                method_frame, header_frame, body = self.channel.basic_get(queue=backward_queue_name, auto_ack=True)

                if method_frame and body:
                    received_data = pickle.loads(body)
                    gradient_numpy = received_data["data"]
                    idx = received_data["idx"]
                    gradient = torch.tensor(gradient_numpy)
                    list_grad[idx] = gradient
                    count += 1
                    if count == size_store:
                        break
            for tensor_grad in list_grad:
                grad_store = torch.cat((grad_store, tensor_grad), dim=0)

            model.train()
            optimizer.zero_grad()
            data_store = data_store.to(self.device)
            out = model(data_store)
            grad_store = grad_store.to(self.device)

            out.backward(gradient=grad_store)
            optimizer.step()

            size_store = 0

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

    def train_on_last_layer(self, model, lr, momentum, clip_grad_norm):

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
        training_flag = False
        time_out = time.time()

        while True:
            method_frame, header_frame, body = self.channel.basic_get(queue=forward_queue_name, auto_ack=True)
            if method_frame and body:

                training_flag = True
                received_data = pickle.loads(body)
                intermediate_output_numpy = received_data["data"]
                labels_numpy = received_data["label"]
                trace = received_data["trace"]
                data_idx = received_data["data_idx"]

                labels = torch.tensor(labels_numpy)
                intermediate_output = torch.tensor(intermediate_output_numpy, requires_grad=True)

                infor_data.append((data_idx, trace, intermediate_output.size(0)))
                tensor_data = torch.cat((tensor_data, intermediate_output), dim=0)
                tensor_label = torch.cat((tensor_label, labels), dim=0)

                time_out = time.time()
                self.data_count += 1
            else:
                if (time.time() - time_out) > 2 and training_flag is True:
                    training_flag = False
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
                    if clip_grad_norm and clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

                    optimizer.step()
                    gradient = tensor_data.grad

                    for (idx, trace, size) in infor_data:
                        grad, new_gradient = gradient.split([size, gradient.size(0) - size], dim=0)
                        gradient = new_gradient

                        self.send_gradient(idx, grad, trace)

                    infor_data = []
                    tensor_data = torch.tensor([])
                    tensor_label = torch.tensor([])

                if training_flag is False:
                    broadcast_queue_name = f'reply_{self.client_id}'
                    method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)
                    if body:
                        received_data = pickle.loads(body)
                        src.Log.print_with_color(f"[<<<] Received message from server {received_data}", "blue")
                        if received_data["action"] == "PAUSE":
                            return result

    def train_on_device(self, model, lr, momentum, clip_grad_norm, train_loader=None, config_time=None,
                        local_round=None):
        self.data_count = 0
        if self.layer_id == 1:
            result = self.train_on_first_layer(model, lr, momentum, clip_grad_norm, train_loader, config_time,
                                               local_round)
        else:
            result = self.train_on_last_layer(model, lr, momentum, clip_grad_norm)

        return result, self.data_count