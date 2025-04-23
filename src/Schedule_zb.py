import time
import uuid
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

from src.Utils import manual_W, hook_model
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
                             control_count=5, train_loader=None, cluster=None, special=False):
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        data_iter = iter(train_loader)
        num_forward = 0
        num_backward = 0
        end_data = False
        micro_batch = 8
        data_store = {}

        backward_queue_name = f"gradient_queue_{self.client_id}"
        self.channel.queue_declare(queue=backward_queue_name, durable=False)
        self.channel.basic_qos(perfetch_count=1)

        model = model.to(self.device)
        neural_layers, inputs_per_layer, outputs_per_layer = hook_model(model)
        storage_to_calculate_W = []
        chunk = 0
        count_w = 0

        with tqdm(total=len(train_loader), desc="Processing", unit="step") as pbar:
            while True:
                # Train model
                model.train()
                optimizer.zero_grad()

                method_frame, header_frame, body = self.channel.basic_get(queue=backward_queue_name, auto_ack=True)
                if method_frame and body:
                    # B process
                    num_backward += 1
                    received_data = pickle.loads(body)
                    grad_output = received_data["data"]
                    grad_output = torch.tensor(grad_output).to(self.device)
                    data_id = received_data["data_id"]

                    load_data = data_store.pop(data_id)
                    torch.autograd.grad(load_data[1], load_data[0], grad_outputs=grad_output, retain_graph=True)
                    grad_of_outputs_per_layer = [_.grad for _ in outputs_per_layer]
                    outputs_per_layer = []
                    storage_to_calculate_W.append([inputs_per_layer, grad_of_outputs_per_layer])
                else:
                    if (len(data_store) <= control_count) and (chunk < micro_batch):
                        # F process
                        try:
                            data, label = next(data_iter)
                            data = data.to(self.device)
                            data_id = uuid.uuid4()
                            output = model(data)
                            load_data.append([data, output])
                            intermediate_output = output.detach().requires_grad_(True)

                            num_forward += 1
                            chunk += 1
                            self.data_count += 1

                            self.send_intermediate_output(data_id, label_count, intermediate_output, label, trace=None, cluster=cluster, special=special)
                        except StopIteration:
                            end_data = True
                    else:
                        # W process
                        if len(storage_to_calculate_W) > 0:
                            load_w = storage_to_calculate_W.pop(0)
                            # load_w : [inputs_per_layer, grad_of_outputs_per_layer]
                            manual_W(load_w[0], load_w[1], neural_layers)  # calculate grad of parameter per layer and sum update it
                            count_w += 1
                            pbar.update(1)
                        else:
                            if count_w == micro_batch:
                                count_w = 0
                                chunk = 0
                                optimizer.step()

