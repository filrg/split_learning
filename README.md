# Split Learning

![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![RabbitMQ](https://img.shields.io/badge/RabbitMQ-FF6600?style=for-the-badge&logo=rabbitmq&logoColor=white)

**Split Learning** is a distributed deep learning training framework where a deep neural network (DNN) model is partitioned into multiple parts and trained across a server and multiple client devices without sharing raw data or the entire model. This ensures data security and privacy, especially for resource-constrained edge devices.

This repository implements **Cluster SFL** (Split Federated Learning), a framework that optimizes split learning across heterogeneous devices by clustering clients with similar data distributions and capabilities, filtering out slow/unreliable devices (stragglers), and adaptively adjusting partition points (cut-layers) for each cluster.

---

## Key Features & Framework Components

**Cluster SFL** addresses issues such as Non-IID data distribution, device heterogeneity, and communication overhead through three core components:

1. **Privacy-Preserving Client Clustering**: Clients' local data distributions are compressed and represented using a Privacy-Preserving Variational Autoencoder (VAE) before clustering, optimizing training performance on Non-IID datasets without exposing raw local data.
2. **GMM-Based Client Selection**: Outliers and slow devices (stragglers) are identified and excluded from the training round using a Gaussian Mixture Model (GMM) that analyzes client computation and communication speeds.
3. **Adaptive Partitioning**: For each client cluster, the framework automatically identifies the optimal model partition point (cut-layer) to balance client-side computation costs and communication overhead.

### Three Execution Modes:
- **No-Cluster Mode (`server.auto-mode: False` & `server.manual.cluster-mode: False`)**: Uses a single static partition point defined globally in `server.manual.no-cluster.cut-layers`.
- **Manual Cluster Mode (`server.auto-mode: False` & `server.manual.cluster-mode: True`)**: Performs manual clustering based on static assignment in `server.manual.cluster`.
- **Auto Cluster Mode (`server.auto-mode: True`)**: Performs dynamic client clustering using KMeans, optional GMM-based client selection, and adaptive partition point computation per cluster.

---

## System Architecture

### Deployment Model
The system consists of a central coordinator (Server) and multiple training nodes (Clients). They communicate asynchronously over RabbitMQ queues.

![deploy_model](pics/deploy_model.png)

To run the system:
1. All server and client processes must connect to the same **RabbitMQ server**.
2. The **Server** manages the global training process, clusters clients, executes the server portion of the model, aggregates weights, and coordinates optimization.
3. The **Clients** load the client portion of the model, process local data to compute forward activations (smashed data), send them to the server, and receive gradients back to compute backward passes and update local weights.

### Split Learning Client Training Flow
Below is the message-based communication cycle between the client and server during training:

![sl_model](pics/sl_model.png)

---

## Supported Models & Datasets

The framework supports multiple DNN models across various modalities:

| Model Name | Dataset | Task Type | Key Modules / Layers |
| :--- | :--- | :--- | :--- |
| **VGG16** | `CIFAR10` | Image Classification | Convolutional, Linear |
| **KWT**  | `SPEECHCOMMANDS` | Audio Classification | Audio processing, Transformer |
| **BERT** | `AGNEWS` | Text Classification | NLP Transformer, LoRA / PEFT |

---

## Installation & Setup

### 1. Python Prerequisites
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```
*Note: If you are training BERT, you also need to install `peft` and `transformers`:*
```bash
pip install peft transformers
```

### 2. RabbitMQ Broker
Setup a RabbitMQ server. A `docker-compose.yaml` is provided to run it easily:
```yaml
version: '3'

services:
  rabbitmq:
    image: rabbitmq:management
    container_name: rabbitmq
    ports:
      - "5672:5672"   # RabbitMQ main port
      - "15672:15672" # Management UI
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: admin
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
volumes:
  rabbitmq_data:
    driver: local
```

Start the RabbitMQ container:
```bash
docker-compose up -d
```
Access the Management UI at `http://localhost:15672` (Credentials: `admin`/`admin` by default).

---

## Step-by-Step Quick Start

Follow these steps in order to start training.

### Step 1: Run Profiling on Clients 
Before running a client, you **MUST** profile its performance. The client profiles its computation speed (execution time per layer) and network latency, writing these metrics to `profiling.json`. If this file is missing, the client cannot register with the server.

Run the profiling tool on each client machine:
```bash
python profiling.py --model KWT --size 32
```
Arguments:
- `--model`: The target model (`VGG16`, `KWT`, or `BERT`).
- `--size` *(Optional)*: Batch size used for profiling (e.g., `32` or `4` for BERT).

### Step 2: Configure the System (`config.yaml`)
Edit the configuration settings in `config.yaml`. (See the **Configuration Reference** section below).

### Step 3: Start the Server
Run the central server:
```bash
python server.py
```
*Note: The server will automatically clean up stale queues from previous runs on startup.*

### Step 4: Start the Clients
Run the clients. Make sure the total number of clients matches what is configured in `config.yaml`.

```bash
python client.py --layer_id 1 --device cuda --cluster 0
```
Arguments:
- `--layer_id`: The client split layer index (starts from `1`).
- `--device` *(Optional)*: Hardware device to use (`cpu`, `cuda`, or `cuda:0`). If omitted, PyTorch will automatically use CUDA if available.
- `--cluster` *(Optional)*: Manually assign the client to a specific cluster ID (starts from `0`). Only relevant if auto-clustering is disabled.

---

## Configuration Reference (`config.yaml`)

Below is the annotation for the version 3.0.0 `config.yaml` schema:

```yaml
name: Split Learning

server:
  global-round: 10 # number of training rounds
  
  clients: # number of clients per layer (e.g. [1, 1] means 1 client in layer 1, 1 client in layer 2)
    - 1
    - 1
    
  auto-mode: False # True to enable auto-clustering (KMeans) and client selection, False to use manual/no-cluster mode
  
  model: KWT               # target model: VGG16, KWT, BERT
  data-name: SPEECHCOMMANDS # target dataset: CIFAR10, SPEECHCOMMANDS, AGNEWS
  
  parameters:
    load: True  # load saved weights (.pth) if exist on server startup
    save: True  # save aggregated weights (.pth) after each round
    
  validation: True # run evaluation on server
  
  data-distribution:
    non-iid: False      # enable non-IID partitioning
    num-sample: 5000    # samples per client
    num-label: 12       # classes in dataset (e.g. 10 for CIFAR10, 12 for SPEECHCOMMANDS, 4 for AGNEWS)
    dirichlet:
      alpha: 1          # Dirichlet alpha parameter for non-IID data
    refresh: True       # regenerate data indices each round

  random-seed: 1
  
  manual:
    cluster-mode: False # True to enable manual cluster configuration, False to run without clustering
    no-cluster:
      cut-layers: [1] # split layer index for no-cluster mode
    cluster:
      num-cluster: 1
      cut-layers:
        - [1] # cut-points for each cluster
      infor-cluster:
        - [1,1] # [layer-1, layer-2] clients per cluster

  cluster-selection:
    num-cluster: 1 # number of clusters for KMeans in auto-mode
    algorithm-cluster: KMeans # clustering algorithm (only KMeans is supported in the main codebase)
    selection-mode: False # True to enable GMM-based client selection to filter out stragglers

rabbit:
  address: 127.0.0.1
  username: admin
  password: admin
  virtual-host: / # RabbitMQ virtual host

log_path: .
debug_mode: True

learning:
  learning-rate: 0.0005
  weight-decay: 0.01 # weight decay (L2 penalty)
  momentum: 0.5
  batch-size: 32
  control-count: 3
```

---

## Model Parameters (`.pth` Files)

After completing a training round, the server aggregates client weights and saves the model checkpoint as `.pth` files in the main directory.
- If `parameters.load` is `True` and the files exist, they will be loaded on startup to resume training.
- To reset or start training from scratch, set `parameters.load` to `False` or delete the `.pth` files.

---

## Baselines (other/)

The repository includes baseline algorithms for benchmarking in the `other/` directory:

| Directory | Baseline Name | Architecture Description |
| :--- | :--- | :--- |
| `Vanilla_SL` | Parallel Split Learning | Standard parallel Split Learning. Static partition, no clustering, all clients train sequentially or concurrently. |
| `Cluster_FSL` | Cluster Split Federated Learning | Sequential training per cluster with manual clustering. No adaptive partitioning. |
| `DCSL` | Data-Center Split Learning | Batch-synchronous training with step-based Learning Rate decay and Server-side Data Aggregation (SDA). |
| `2LS` | Two-Layer Split Learning | Hierarchical 2-stage clustering (outer/inner), async communication (FedAsync), and SDA at the server. |
| `FLEX` | FLEX Framework | Dynamic clustering using label distributions, GMM client selection, and adaptive cut-layer partitioning. |

---

## Publication & Reference

If you use this framework in your research, please cite the original paper:

```text
Efficient Split Federated Learning via Privacy-Preserving Client Clustering, Resource-Aware Client Selection, and Adaptive Partitioning
(IEEE Access)
```

---

## TODOs & Roadmap
- [ ] Add inference mode interface separate from training flow.
- [ ] Implement automatic RabbitMQ queue deletion on program finish/exit.
- [ ] Implement close connection requests to clean up connections gracefully.
