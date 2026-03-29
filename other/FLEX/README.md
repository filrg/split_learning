# FLEX

### Server
```commandline
python3 server.py
```
### Client 
#### DAI
```commandline
python3.8 client.py --layer_id 2 --c 0 --s
python3 client.py --layer_id 2 --c 1 --s
python3 client.py --layer_id 2 --c 2 --s
```
machine 12, 3, 4 (reject 1 device)
```commandline
python3 client.py --layer_id 1 --c 0 --s 
python3 client.py --layer_id 1 --c 0 --s 
python3 client.py --layer_id 1 --c 0 
```
machine 5 6 7 (reject 1 device)
```commandline
python3 client.py --layer_id 1 --c 1 --s 
python3 client.py --layer_id 1 --c 1 --s 
python3 client.py --layer_id 1 --c 1 
```
machine 8 9 10 (reject 1 device)
```commandline
python3 client.py --layer_id 1 --c 2 --s 
python3 client.py --layer_id 1 --c 2 --s 
python3 client.py --layer_id 1 --c 2 
```

* ``layer_id`` : ID of layer, start from 1
* ``c``: ID cluster by device
* ``s``: Select (have --s) /Reject device (don't have --s) 

