# How to run 2Ls

## SERVER

```commandline
python3 server.py
```

## Client
### dai
```commandline
python3 client.py --layer_id 2 --idx 0  --incluster 0
python3 client.py --layer_id 2 --idx 1  --incluster 0
python3 client.py --layer_id 2 --idx 2  --incluster 1
```
machine 12, 3, 8
```commandline
python3.8 client.py --layer_id 1 --idx 0 --incluster 0 --outcluster 0
python3 client.py --layer_id 1 --idx 1 --incluster 0 --outcluster 0
python3 client.py --layer_id 1 --idx 2 --incluster 1 --outcluster 0
```
machine 4, 5, 9
```commandline
python3 client.py --layer_id 1 --idx 0 --incluster 0 --outcluster 1
python3 client.py --layer_id 1 --idx 1 --incluster 0 --outcluster 1
python3 client.py --layer_id 1 --idx 2 --incluster 1 --outcluster 1
```
machine 6, 7, 10
```commandline
python3 client.py --layer_id 1 --idx 0 --incluster 0 --outcluster 2
python3 client.py --layer_id 1 --idx 1 --incluster 0 --outcluster 2
python3 client.py --layer_id 1 --idx 2 --incluster 1 --outcluster 2
```
