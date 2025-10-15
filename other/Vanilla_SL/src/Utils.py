import numpy as np
import random
import pika
import torch

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


def fedavg_state_dicts(state_dicts, weights = None):
    """
    Trung bình (FedAvg) một list các state_dict.
    - state_dicts: list các dict {param_name: tensor}
    - weights: list trọng số tương ứng (mặc định None nghĩa là mỗi model weight=1)
    Trả về một dict {param_name: tensor_avg}
    """
    num = len(state_dicts)
    if num == 0:
        raise ValueError("fedavg_state_dicts: không có state_dict nào để trung bình.")

    if weights is None:
        weights = [1.0] * num
    total_w = sum(weights)

    # Tập hợp tất cả key
    all_keys = set().union(*(sd.keys() for sd in state_dicts))
    avg_dict = {}

    for key in all_keys:
        # gom tensor + weight, xử lý NaN
        acc = None
        for sd, w in zip(state_dicts, weights):
            if key not in sd:
                continue
            t = sd[key].float()
            if torch.isnan(t).any():
                t = torch.nan_to_num(t)  # zero-fill
            t = t * w
            acc = t if acc is None else acc + t

        # chia trung bình
        avg = acc / total_w

        # cast về dtype gốc
        orig = next(sd[key] for sd in state_dicts if key in sd)
        if orig.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.bool):
            avg = avg.round().to(orig.dtype)
        else:
            avg = avg.to(orig.dtype)

        avg_dict[key] = avg

    return avg_dict
