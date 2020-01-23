# MSG-GAN benchmark using TensorFlowV2 Strategy and Fasion-MNIST dataset

Last time we checked if it still works: ![](https://img.shields.io/date/1579778719) ![](https://img.shields.io/badge/platform-x86_64-blue.svg)


This example, based on the [TensorFlow 2.0 in Action blogpost](https://blog.paperspace.com/tensorflow-2-0-in-practice/),
utilizes the [tf.distribute.Strategy] API. The example is easy to run, because it uses the Fashion-MNIST dataset easily
available through keras API. At the same time it has benchmarking potential, as it trains a model belonging to the
computationally expensive GAN class.


## Table of contents
- [x] [Instructions for running the benchmarks](#running-the-benchmarks)
- [ ] [Experimental results](#experimental-results)
    - [ ] MirroredStrategy
    - [x] [MultiWorkerMirroredStrategy](#multiworkermirroredstrategy)
    - [ ] TPUStrategy - help wanted

## Running the benchmarks

### Prerequisites

* GNU/Linux
* Python 3.6, Pip, git

### Installation

**Clone this repository**
```bash
git clone https://github.com/roscisz/dnn_training_benchmarks.git
cd dnn_training_benchmarks/TensorFlowV2_MSG-GAN_Fashion-MNIST/
```

**Install requirements**
```bash
pip install -r requirements.txt
```

### Running

**Single node**
```bash
python train.py
```

**Multi-node**
On each node, a separate process has to be run with the appropriate TF_CONFIG setting.
For example, a training on two nodes gl01 and gl02 would require running the following
processes:

gl01:
```bash
TF_CONFIG='{"cluster":{"worker":["gl01:2222", "gl02:2222"]}, "task":{"type": "worker", "index": 0}}' python train.py
```

gl02:
```bash
TF_CONFIG='{"cluster":{"worker":["gl01:2222", "gl02:2222"]}, "task":{"type": "worker", "index": 1}}' python train.py
```

In our experiments, we used the [TensorHive](https://github.com/roscisz/TensorHive) tool for easier
orchestration of the tasks on multiple nodes, as presented [here](https://github.com/roscisz/TensorHive/tree/master/examples/TF_CONFIG).


## Experimental results

### MultiWorkerMirroredStrategy

In this section we present performance results of the benchmark application
on a cluster with 16 nodes, each equipped with two Intel(R) Xeon(R) CPU E5-2670 v3 @ 2.30GHz
CPUs.

#### Multi-node scalability
![scalability32](https://raw.githubusercontent.com/roscisz/dnn_training_benchmarks/master/TensorFlowV2_MSG-GAN_Fashion-MNIST/img/scalability32.png)
