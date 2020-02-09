
# TensorFlow MNIST CNN

This example shows distributed training of a simple convolutional model using the MNIST dataset.
Two approaches to distributed trainig are presented:
* multiple GPUs on a single node
* multiple nodes (each with potentially multiple GPUs)

Both examples use [**tf.distribute.Strategy**](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy) module, which is a part of **TensorFlow 2.0** and above.
### Prerequisites

* GNU/Linux
* Python 3.6, Pip, git

### Installation

**Clone this repository**
```bash
git clone https://github.com/roscisz/dnn_training_benchmarks.git
cd dnn_training_benchmarks/TensorFlowV2_CNN_MNIST/
```
**Install requirements**
```bash
pip install -r requirements.txt
```
### Trainig parameters
While running each of the example scripts you can adjust training settings by setting command line paramters:

 - epochs
 - batch_size
 - steps_per_epoch

Example:
```bash
 python train.py --batch_size=64 --epochs=5 --steps_per_epoch=500
```
## Single GPU
Simple implementation, which will be used as a base for distributed examples.

Code: **train.py**

## Multiple GPUs on a single node
It is actually very easy to run TensorFlow code on multiple GPUs. 
All we need to do is to specify an appropriate strategy:
```python
strategy = tf.distribute.MirroredStrategy()
```
And then define and compile our model within the strategy scope:
```python
with strategy.scope():
    model = build_and_compile_cnn_model()
```
That is all we have to do. Now, when started, training will be distributed on all available GPUs.

For further information on **MirroredStrategy** please check the official TensorFlow distributed training [tutorial](https://www.tensorflow.org/guide/distributed_training).

Code: **train_mirrored_strategy.py**

## Multiple nodes
Running our code on multiple nodes is just a bit more complicated. First of all, we need to use the **MultiWorkerMirroredStrategy** instead of the **MirroredStrategy**:
```python
strategy = tf.distribute.MultiWorkerMirroredStrategy()
```

```python
with strategy.scope():
    model = build_and_compile_cnn_model()
```

For distributed training we need to run the script on all nodes. But, before we start training, thre is one more thing to be done. On each node we need to specify the **TF_CONFIG** environment variable, which is responsible for configuration of distributed training in TensorFlow. **TF_CONFIG** example:
```json
{ 
   "cluster":{ 
      "worker":[ 
         "172.16.0.1:12345",
         "172.16.0.2:12345"
      ]
   },
   "task":{ 
      "type":"worker",
      "index":0
   }
}
```
There are two components of **TF_CONFIG**: **cluster** and **task**. **cluster** provides information about the whole training cluster, which is a dict consisting of different types of jobs such as worker. So all nodes will have the same value of **cluster** component. **task** on the other hand provides information of the current task, so its value will be different on each node. For example: if we are running our script on two nodes, one of them should have index=0 and the other index=1.

For further information on **MultiWorkerMirroredStrategy** distributed training please check the official TensorFlow [tutorial](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras).

Code: **train_multi_worker_mirrored_strategy.py**
