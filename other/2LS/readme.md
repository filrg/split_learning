# SERVER

```
python3 server.py

# SPLIT SERVER (Layer 2)
python3 client.py --layer_id 2 --idx 0 --incluster 0 --outcluster 0
python3 client.py --layer_id 2 --idx 1 --incluster 0 --outcluster 0
python3 client.py --layer_id 2 --idx 2 --incluster 1 --outcluster 0

# OUT-CLUSTER 0 - Layer 1
python3 client.py --layer_id 1 --idx 0 --incluster 0 --outcluster 0
python3 client.py --layer_id 1 --idx 1 --incluster 0 --outcluster 0
python3 client.py --layer_id 1 --idx 2 --incluster 1 --outcluster 0


# OUT-CLUSTER 1 - Layer 1
python3 client.py --layer_id 1 --idx 0 --incluster 0 --outcluster 1
python3 client.py --layer_id 1 --idx 1 --incluster 0 --outcluster 1
python3 client.py --layer_id 1 --idx 2 --incluster 1 --outcluster 1

# OUT-CLUSTER 2 - Layer 1
python3 client.py --layer_id 1 --idx 0 --incluster 0 --outcluster 2
python3 client.py --layer_id 1 --idx 1 --incluster 0 --outcluster 2
python3 client.py --layer_id 1 --idx 2 --incluster 1 --outcluster 2