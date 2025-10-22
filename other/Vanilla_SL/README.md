# Introduce to run Vannila SL
## How to Run
### 1. Run server
First, run a server to control all training devices and aggregate models.

```commandline
python server.py
```
### 2. Run client
In this case, devices will be trained sequentially.
```commandline
python client.py --layer_id 1
```
- `layer_id`: is the ID index of client's layer, start from 1.

## Information training
### Devices
- **Layer1** have 5 machines and 4 jetson nano. 

- **Layer2** have 3 DAI. 
### Model and Dataset
- VGG16
- CIFAR10 with 2500 or 25 images in each device.  (two cases)
```yaml
  data-distribution:
    non-iid: False
    num-sample: 2500 #(line 23)
    num-label: 10
    dirichlet:
      alpha: 1
    refresh-each-round: True
  random-seed: 1
```
- In case of non-iid, we change function `distribution` in `Server.py`. We will manually configure label distribution on each device.
> **Warning: Synchronize data distribution between instances**
```python
def distribution(self):
    if self.non_iid:
        label_distribution = np.random.dirichlet([self.data_distribution["dirichlet"]["alpha"]] * self.num_label,
                                                 self.total_clients[0])

        # Change here (line 90)
        self.label_counts = (label_distribution * self.num_sample).astype(int)
    else:
        self.label_counts = np.full((self.total_clients[0], self.num_label), self.num_sample // self.num_label)
```

### Round training

- **Global_round** = 50
- Always unlimited time. (enable = False)

## Choose cut point
When training, we will measure the cut-off points of 7, 14, and 24. We will change it in the `config.yaml`.
```yaml
  local-round: 1
  global-round: 1
  clients:
    - 1
    - 1
  no-cluster:
    cut-layers: [7] # fix here (line 13)
```
