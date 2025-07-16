import time
import uuid
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

from src.Utils import manual_W
import src.Log


class Scheduler:
    def __init__(self, client_id, layer_id, channel, device):
        self.client_id = client_id
        self.layer_id = layer_id
        self.channel = channel
        self.device = device
        self.data_count = 0

    def balanced_softmax_loss(self, logits, labels, class_counts, epsilon=1e-6):
        class_counts = torch.tensor(class_counts, dtype=torch.int64).to(self.device)
        log_probs = f.log_softmax(logits, dim=1)
        class_probs = class_counts / (class_counts.sum() + epsilon)
        weights = 1.0 / (class_probs + epsilon)
        weights = weights / weights.sum()
        loss = (-weights[labels] * log_probs[range(labels.shape[0]), labels]).mean()
        return loss

    def send_intermediate_output(self, data_id, label_count, output, labels, trace, cluster=None, special=False):
        if special:
            forward_queue_name = f"intermediate_queue_{self.layer_id}"
        else:
            forward_queue_name = f"intermediate_queue_{self.layer_id}_{cluster}"
        self.channel.queue_declare(forward_queue_name, durable=False)

        if trace:
            trace.append(self.client_id)
            message = pickle.dumps(
                {"data_id": data_id, "label_count": label_count, "data": output.detach().cpu().numpy(),
                 "labels": labels, "trace": trace}
            )
        else:
            message = pickle.dumps(
                {"data_id": data_id, "label_count": label_count, "data": output.detach().cpu().numpy(),
                 "labels": labels, "trace": [self.client_id]}
            )

        self.channel.basic_publish(
            exchange='',
            routing_key=forward_queue_name,
            body=message
        )

    def send_gradient(self, data_id, gradient, trace):
        to_client_id = trace[-1]
        trace.pop(-1)
        backward_queue_name = f'gradient_queue_{to_client_id}'
        self.channel.queue_declare(backward_queue_name, durable=False)

        message = pickle.dumps(
            {"data_id": data_id, "data": gradient.detach().cpu().numpy(), "trace": trace}
        )

        self.channel.basic_publish(
            exchange='',
            routing_key=backward_queue_name,
            body=message
        )

    def send_to_server(self, message):
        self.channel.queue_declare('rpc_queue', durable=False)
        self.channel.basic_publish(
            exchange='',
            routing_key='rpc_queue',
            body=pickle.dumps(message)
        )

    def train_on_first_layer(self, model, global_model, label_count, lr, momentum, clip_grad_norm, computer_loss,
                             control_count=5, train_loader=None, cluster=None, special=False, chunks=1):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        data_iter = iter(train_loader)
        model = model.to(self.device)
        num_forward = 0
        num_backward = 0
        num_weight = 0
        end_data = False
        data_store = {}
        dict_outputs_per_layer = {}
        inputs_per_layer = []
        outputs_per_layer = []
        neural_layers = []
        storage_to_calculate_W = []
        def check_layer(the_layer):
           return isinstance(the_layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d))
        def save_output_hook(module, the_input, the_output):
            if check_layer(module):
                inputs_per_layer.append(the_input[0].detach())
                the_output.retain_grad()
                outputs_per_layer.append(the_output)

        for layer in model.modules():
            if check_layer(layer):
                neural_layers.append(layer)
                layer.register_forward_hook(save_output_hook)

        backward_queue_name = f"gradient_queue_{self.client_id}"
        self.channel.queue_declare(queue=backward_queue_name, durable=False)
        self.channel.basic_qos(prefetch_count=1)

        with tqdm(total=len(train_loader), desc="Processing", unit="step") as pbar:
            while True:
                try:
                    images , labels = next(data_iter)
                    image_chunks = torch.chunk(images, chunks, dim=0)
                    label_chunks = torch.chunk(labels, chunks, dim=0)
                    micro_batch = iter(zip(image_chunks, label_chunks))
                    end_batch = False
                    model.train()
                    optimizer.zero_grad()
                    # Train model
                    while True:
                        # Check_backward
                        method_frame, header_frame, body = self.channel.basic_get(queue=backward_queue_name, auto_ack=True)
                        if method_frame and body:
                            # B process
                            num_backward += 1
                            # Read message
                            received_data = pickle.loads(body)
                            grad_output = received_data["data"]
                            grad_output = torch.tensor(grad_output).to(self.device)
                            data_id = received_data["data_id"]

                            load_data = data_store.pop(data_id)
                            # Calculate gradient loss from x
                            grad_x = torch.autograd.grad(load_data[1], load_data[0], grad_outputs=grad_output, retain_graph=True)[0]
                            # take out grad of each layer in model
                            grad_of_outputs_per_layer = [_.grad for _ in dict_outputs_per_layer[data_id]]
                            # clear
                            dict_outputs_per_layer[data_id].clear()
                            del dict_outputs_per_layer[data_id]
                            storage_to_calculate_W.append([load_data[2], grad_of_outputs_per_layer])
                        else:
                            if not end_batch:
                                # F process
                                try:
                                    data, label = next(micro_batch)
                                    data = data.to(self.device).requires_grad_()

                                    data_id = uuid.uuid4()
                                    output = model(data)
                                    output.retain_grad()
                                    data_store[data_id] = [data, output, inputs_per_layer]
                                    inputs_per_layer = []
                                    dict_outputs_per_layer[data_id] = outputs_per_layer
                                    outputs_per_layer = []
                                    intermediate_output = output.detach().requires_grad_(True)

                                    num_forward += 1
                                    self.data_count += 1

                                    self.send_intermediate_output(data_id, label_count, intermediate_output, label, trace=None, cluster=cluster, special=special)
                                except StopIteration:
                                    end_batch = True
                            else:
                                # W process
                                if len(storage_to_calculate_W) > 0:
                                    load_w = storage_to_calculate_W.pop(0)
                                    # load_w : [inputs_per_layer, grad_of_outputs_per_layer]
                                    manual_W(load_w[0], load_w[1], neural_layers)  # calculate grad of parameter per layer and sum update it
                                    num_weight += 1

                        if (num_forward == num_backward == num_weight) and end_batch:
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)
                            num_forward = 0
                            num_backward = 0
                            num_weight = 0
                            break
                except StopIteration:
                    end_data = True
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


    def train_on_last_layer(self, model, global_model, label_count, lr, momentum, clip_grad_norm, compute_loss, cluster,
                            special=False, chunks=1):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        optimizer.zero_grad()
        result = True

        criterion = nn.CrossEntropyLoss()
        if special:
            forward_queue_name = f'intermediate_queue_{self.layer_id - 1}'
        else:
            forward_queue_name = f'intermediate_queue_{self.layer_id - 1}_{cluster}'
        self.channel.queue_declare(queue=forward_queue_name, durable=False)
        self.channel.basic_qos(prefetch_count=1)
        print('Waiting for intermediate output. To exit press CTRL+C')
        model.to(self.device)
        inputs_per_layer = []
        outputs_per_layer = []
        neural_layers = []

        def check_layer(the_layer):
            return isinstance(the_layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d))

        def save_output_hook(module, the_input, the_output):
            if check_layer(module):
                inputs_per_layer.append(the_input[0].detach())
                the_output.retain_grad()
                outputs_per_layer.append(the_output)

        for layer in model.modules():
            if check_layer(layer):
                neural_layers.append(layer)
                layer.register_forward_hook(save_output_hook)

        num_forward = 0
        num_backward = 0
        num_weight = 0
        storage_to_calculate_W = []
        while True:
            if num_forward < chunks:
                method_frame, header_frame, body = self.channel.basic_get(queue=forward_queue_name, auto_ack=True)
                if method_frame and body:
                    received_data = pickle.loads(body)
                    intermediate_output_numpy = received_data['data']
                    trace = received_data['trace']
                    data_id = received_data['data_id']
                    labels = received_data['labels'].to(self.device)
                    label_count = received_data['label_count']

                    intermediate_output = torch.tensor(intermediate_output_numpy, requires_grad=True).to(self.device)
                    intermediate_output.retain_grad()

                    # F process
                    output = model(intermediate_output)
                    num_forward += 1
                    self.data_count += 1
                    loss = criterion(output, labels)
                    print(f'Loss: {loss.item()}')
                    if torch.isnan(loss).any():
                        src.Log.print_with_color("NaN detected in loss", 'yellow')
                        result = False

                    # B process
                    grad_out = torch.autograd.grad(loss, output, retain_graph=True)[0]
                    gradient_intermediate =  torch.autograd.grad(output, intermediate_output, grad_outputs=grad_out, retain_graph=True)[0]
                    self.send_gradient(data_id, gradient_intermediate, trace)
                    grads_of_output_per_layer = [_.grad for _ in outputs_per_layer]
                    outputs_per_layer = []
                    storage_to_calculate_W.append([inputs_per_layer, grads_of_output_per_layer])
                    num_backward += 1
                    inputs_per_layer = []

                else:
                    # W process
                    if len(storage_to_calculate_W) > 0:
                        load_w = storage_to_calculate_W.pop(0)
                        # load_w : [inputs_per_layer, grad_of_outputs_per_layer]
                        manual_W(load_w[0], load_w[1], neural_layers)  # calculate grad of parameter per layer and sum update it
                        num_weight += 1

                    else:
                        # Check PAUSE from server
                        if num_forward == num_backward == num_weight:
                            broadcast_queue_name = f'reply_{self.client_id}'
                            method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)

                            if body:
                                received_data = pickle.loads(body)
                                src.Log.print_with_color(f'[<<<] Received message from server {received_data}', 'blue')
                                if received_data['action'] == 'PAUSE':
                                    optimizer.step()
                                    optimizer.zero_grad()
                                    return result

            # perform remaining W and otp step
            else:
                if len(storage_to_calculate_W) > 0:
                    load_w = storage_to_calculate_W.pop(0)
                    manual_W(load_w[0], load_w[1], neural_layers)
                    num_weight += 1
                else:
                    if num_forward == num_backward == num_weight:
                        optimizer.step()
                        optimizer.zero_grad()
                        num_forward = 0
                        num_backward = 0
                        num_weight = 0

    def train_on_middle_layer(self, model, global_model, label_count, lr, momentum, clip_grad_norm, compute_loss, control_count=5, cluster=None, special=False, chunks=1):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss()
        result = True
        if special:
            forward_queue_name = f'intermediate_queue_{self.layer_id - 1}'
        else:
            forward_queue_name = f'intermediate_queue_{self.layer_id - 1}_{cluster}'
        backward_queue_name = f'gradient_queue_{self.client_id}'
        self.channel.queue_declare(queue=forward_queue_name, durable=False)
        self.channel.queue_declare(queue=backward_queue_name, durable=False)
        self.channel.basic_qos(prefetch_count=1)
        data_store = {}
        dict_outputs_per_layer = {}
        print('Waiting for intermediate output. To exit press CTRL+C')

        model.to(self.device)
        inputs_per_layer = []
        outputs_per_layer = []
        neural_layers = []
        def check_layer(the_layer):
            return isinstance(the_layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d))

        def save_output_hook(module, the_input, the_output):
            if check_layer(module):
                inputs_per_layer.append(the_input[0].detach())
                the_output.retain_grad()
                outputs_per_layer.append(the_output)

        for layer in model.modules():
            if check_layer(layer):
                neural_layers.append(layer)
                layer.register_forward_hook(save_output_hook)

        num_forward = 0
        num_backward = 0
        num_weight = 0
        storage_to_calculate_W = []

        while True:
            # B process
            method_frame, header_frame, body = self.channel.basic_get(queue=backward_queue_name, auto_ack=True)
            if method_frame and body:
                received_data = pickle.loads(body)
                grad_output = received_data['data']
                grad_output = torch.tensor(grad_output).to(self.device)
                trace = received_data["trace"]
                data_id = received_data["data_id"]

                num_backward += 1
                load_data = data_store.pop(data_id)
                # Calculate gradient loss from x
                grad_x = torch.autograd.grad(load_data[1], load_data[0], grad_outputs=grad_output, retain_graph=True)[0]
                self.send_gradient(data_id, grad_x, trace)
                # take out grad of each layer in model
                grad_of_outputs_per_layer = [_.grad for _ in dict_outputs_per_layer[data_id]]
                dict_outputs_per_layer[data_id].clear()
                del dict_outputs_per_layer[data_id]
                storage_to_calculate_W.append([load_data[2], grad_of_outputs_per_layer])
                continue

            if num_forward < chunks:
                method_frame, header_frame, body = self.channel.basic_get(queue=forward_queue_name, auto_ack=True)
                if method_frame and body:
                    received_data = pickle.loads(body)
                    intermediate_output_numpy = received_data["data"]
                    intermediate_output = torch.tensor(intermediate_output_numpy, requires_grad=True).to(self.device)
                    intermediate_output.retain_grad()
                    trace = received_data["trace"]
                    data_id = received_data["data_id"]
                    labels = received_data["labels"].to(self.device)
                    label_count = received_data["label_count"]

                    # F process
                    output = model(intermediate_output)
                    output.retain_grad()
                    data_store[data_id] = [intermediate_output, output, inputs_per_layer]
                    inputs_per_layer = []
                    dict_outputs_per_layer[data_id] = outputs_per_layer
                    outputs_per_layer = []
                    intermediate_output = output.detach().requires_grad_(True)

                    num_forward += 1
                    self.data_count += 1

                    self.send_intermediate_output(data_id, label_count, intermediate_output, labels, trace=trace, cluster=cluster, special=special)
                    continue

            if len(storage_to_calculate_W) > 0:
                # W process
                load_w = storage_to_calculate_W.pop(0)
                manual_W(load_w[0], load_w[1], neural_layers)
                num_weight += 1
                continue
            else:
                if num_forward == num_backward == num_weight == chunks:
                    optimizer.step()
                    optimizer.zero_grad()
                    num_forward = 0
                    num_backward = 0
                    num_weight = 0
                    continue

            if num_forward == num_backward == num_weight:
                broadcast_queue_name = f'reply_{self.client_id}'
                method_frame, header_frame, body = self.channel.basic_get(queue=broadcast_queue_name, auto_ack=True)

                if body:
                    received_data = pickle.loads(body)
                    src.Log.print_with_color(f'[<<<] Received message from server {received_data}', 'blue')
                    if received_data['action'] == 'PAUSE':
                        for Layer in neural_layers:
                            if Layer.weight.grad is not None:
                                Layer.weight.grad /= chunks
                            if Layer.bias.grad is not None:
                                Layer.bias.grad /= chunks
                        optimizer.step()
                        optimizer.zero_grad()
                        return result

    def alone_training(self, model, global_model, label_count, lr, momentum, clip_grad_norm, compute_loss, train_loader=None, cluster=None, chunks=1):
        return True

    def train_on_device(self, model, global_model, label_count, lr, momentum, clip_grad_norm, compute_loss, num_layers,
                        control_count, train_loader=None, cluster=None, special=False, alone_train=False, chunk=1):
        self.data_count = 0
        if self.layer_id == 1:
            if alone_train is False:
                result = self.train_on_first_layer(model, global_model, label_count, lr, momentum, clip_grad_norm,
                                                   compute_loss, control_count, train_loader, cluster, special, chunks=chunk)
            else:
                result = self.alone_training(model, global_model, label_count, lr, momentum, clip_grad_norm,
                                             compute_loss, train_loader=train_loader, cluster=cluster, chunks=chunk)
        elif self.layer_id == num_layers:
            result = self.train_on_last_layer(model, global_model, label_count, lr, momentum, clip_grad_norm,
                                              compute_loss, cluster=cluster, special=special, chunks=chunk)
        else:
            result = self.train_on_middle_layer(model, global_model, label_count, lr, momentum, clip_grad_norm,
                                                compute_loss, control_count, cluster=cluster, special=special, chunks=chunk)

        return result, self.data_count

# Scheduler_zb: chunks
