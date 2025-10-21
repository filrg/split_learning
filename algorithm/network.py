import pika
import time
import pickle
from tqdm import tqdm

def network(channel, rounds = 100, id_client = None):

    speed_all = []
    size_data = []
    for i in range(1, 10):
        size_data.append(i * 10**6)
    queue_name = f"test_network_{id_client}"
    channel.queue_declare(queue=queue_name, durable=True)

    for size in size_data:
        message = size * '1'
        avg_time = 0.0
        for _ in tqdm(range(rounds)):
            time_stamp = time.time()
            channel.basic_publish(exchange='',
                                  routing_key=queue_name,
                                  body=pickle.dumps(message),
                                  properties=pika.BasicProperties(
                                      expiration='10',  # TTL = 10 milliseconds
                                      delivery_mode=1  # non-persistent (optional)
                                    )
                                  )
            avg_time += ((time.time() - time_stamp) * 10 ** 9)
        avg_time = avg_time / rounds
        speed = size / avg_time
        speed_all.append(speed)
    channel.queue_delete(queue=queue_name)
    speed = sum(speed_all) / len(speed_all)
    speed =  round(speed, 4)

    return speed

