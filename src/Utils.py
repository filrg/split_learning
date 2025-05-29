import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import pika
from requests.auth import HTTPBasicAuth
import requests


def delete_old_queues(address, username, password, virtual_host):
    url = f'http://{address}:15672/api/queues'
    response = requests.get(url, auth=HTTPBasicAuth(username, password))

    if response.status_code == 200:
        queues = response.json()

        credentials = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(pika.ConnectionParameters(address, 5672, f'{virtual_host}', credentials))
        http_channel = connection.channel()

        for queue in queues:
            queue_name = queue['name']
            if queue_name.startswith("reply") or queue_name.startswith("intermediate_queue") or queue_name.startswith(
                    "gradient_queue") or queue_name.startswith("rpc_queue"):

                http_channel.queue_delete(queue=queue_name)

            else:
                http_channel.queue_purge(queue=queue_name)

        connection.close()
        return True
    else:
        return False


def change_state_dict(state_dicts, i):
    def change_name(name):
        parts = name.split(".", 1)
        number = int(parts[0]) + i
        name = f"{number}" + "." + parts[1]
        return name

    new_state_dict = {}
    for key, value in state_dicts.items():
        new_key = change_name(key)
        new_state_dict[new_key] = value
    return new_state_dict


def non_iid_rate(num_data, rate):
    result = []
    for _ in range(num_data):
        if rate < random.random():
            result.append(0)
        else:
            result.append(1)
    return np.array(result)


def num_client_in_cluster(client_cluster_label):
    max_val = max(client_cluster_label)
    count_list = [0] * (max_val + 1)
    for num in client_cluster_label:
        count_list[num] += 1
    count_list = [[x] for x in count_list]
    return count_list


def manual_linear_grad_weight(x, grad_out, linear_layer):
    """
    Calculate manual ∂L/∂W for nn.Linear
    Args:
        x: input of layer, shape [B, in_features]
        grad_out: ∂L/∂z, shape [B, out_features]
        linear_layer: nn.Linear
    Returns:
        grad_w: [out_features, in_features]
    """
    grad_w = grad_out.T @ x  # [out_features, in_features]
    return grad_w


def manual_BatchNorm2d_grad_weight(x_in, grad_out, bn_layer, eps=1e-5):
    """
    Calculate manual ∂L/∂gamma, ∂L/∂beta for BatchNorm2d
    x_in: [B, C, H, W]
    grad_out: ∂L/∂y, [B, C, H, W]
    bn_layer: nn.BatchNorm2d
    """
    mu = x_in.mean(dim=(0, 2, 3), keepdim=True)
    var = x_in.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
    x_hat = (x_in - mu) / torch.sqrt(var + eps)

    grad_gamma = (grad_out * x_hat).sum(dim=(0, 2, 3))
    grad_beta = grad_out.sum(dim=(0, 2, 3))
    return grad_gamma, grad_beta


def manual_conv_grad_weight(x_in, grad_out, conv_layer):
    """
    Calculate manual ∂L/∂W for Conv_2d
    Args:
        x_in: đầu vào của layer này, shape [N, Cin, H, W]
        grad_out: ∂L/∂z (gradient output), shape [N, Cout, H_out, W_out]
        conv_layer: đối tượng nn.Conv2d
    Returns:
        grad_w: ∂L/∂W, shape [Cout, Cin, Kh, Kw]
    """
    Kh, Kw = conv_layer.kernel_size
    stride = conv_layer.stride
    padding = conv_layer.padding
    dilation = conv_layer.dilation

    x_unfold = F.unfold(x_in, kernel_size=(Kh, Kw), stride=stride, padding=padding, dilation=dilation)

    N = grad_out.shape[0]
    grad_out = grad_out.reshape(N, grad_out.shape[1], -1)  # [N, Cout, L]
    grad_w_batch = torch.bmm(grad_out, x_unfold.transpose(1, 2))  # [N, Cout, Cin*Kh*Kw]
    grad_w = grad_w_batch.sum(dim=0).view(conv_layer.out_channels, conv_layer.in_channels, *conv_layer.kernel_size)
    return grad_w


def manual_W(inputs, grads_z_per_layer, neural_layers):
    for i in range(len(neural_layers)):
        x_in = inputs[i]
        grad_out = grads_z_per_layer[i]  # ∂L/∂z tại layer[i]

        Layer = neural_layers[i]
        if isinstance(Layer, nn.Conv2d):
            grad_w = manual_conv_grad_weight(x_in, grad_out, Layer)
            # grad_w = torch.tensor(grad_w)
            if Layer.weight.grad is None:
                Layer.weight.grad = grad_w.clone().detach()
            else:
                Layer.weight.grad += grad_w.clone().detach()
            if Layer.bias.grad is not None:
                # ∂L/∂b = sum over batch, height, width
                grad_b = grad_out.sum(dim=(0, 2, 3))
                Layer.bias.grad += grad_b
            else:
                grad_b = grad_out.sum(dim=(0, 2, 3))
                Layer.bias.grad = grad_b

        elif isinstance(Layer, nn.Linear):
            grad_w = manual_linear_grad_weight(x_in, grad_out, Layer)
            # grad_w = torch.tensor(grad_w)
            if Layer.weight.grad is None:
                Layer.weight.grad = grad_w.clone().detach()
            else:
                Layer.weight.grad += grad_w.clone().detach()
            if Layer.bias.grad is not None:
                # ∂L/∂b = sum over batch
                grad_b = grad_out.sum(dim=0)
                Layer.bias.grad += grad_b
            else:
                grad_b = grad_out.sum(dim=0)
                Layer.bias.grad = grad_b
