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
