# DeepSpeech benchmarks

Last time we checked if it still works: ![](https://img.shields.io/date/1583517314) ![](https://img.shields.io/badge/platform-x86_64-blue.svg)

In this example we provide experimental results and steps to reproduce for benchmarking performance of training the
[Baidu's Deep Speech](https://arxiv.org/abs/1412.5567) Recurrent Neural Network for automatic speech recognition.

The example is based on [Project DeepSpeech by Mozilla](https://github.com/mozilla/DeepSpeech) which is an open source
implementation in TensorFlow that supports distributed training using Distributed TensorFlow.

## Table of contents
- [x] [Installation instructions](#installation)
- [x] [Instructions for running the benchmarks](#running-the-benchmarks)
  - [x] [Manually](#manually)
  - [x] [Using run_cluster.sh](#run-clustersh)
  - [x] [Using Docker](#docker)
  - [x] [Using Kubernetes](#kubernetes)
- [x] [Experimental results](#experimental-results):
  - [x] [Batch size influence on training performance on various GPUs](#batch-size)
  - [x] [Scalability on multiple GPUs](#multigpu-scalability)

## Installation

In this section we describe installation steps for DeepSpeech that we used in our setup.
For detailed instructions for running the DeepSpeech training go to the
[DeepSpeech project site](https://github.com/mozilla/DeepSpeech).

### Prerequisites

* GNU/Linux
* Python 3.6, Pip, git, wget
* CUDA 9.0 with CuDNN 7
* TensorFlow 1.6.0
* Python packages: pandas, python_speech_features, pyxdg, progressbar2

The environment can be for example set up using nvidia-docker as follows:

```bash
nvidia-docker pull nvidia/cuda:9.0-cudnn7-devel
nvidia-docker run -it nvidia/cuda:9.0-cudnn7-devel
apt-get update
apt-get install -y git wget software-properties-common virtualenv
add-apt-repository ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.6 python3-pip
virtualenv -p python3.6 venv
source venv/bin/activate
pip install 'tensorflow-gpu==1.6.0' 'pandas==0.25.1' python_speech_features pyxdg progressbar2 scipy
```

### Installing DeepSpeech

**Clone the proper version of DeepSpeech**
```bash
git clone https://github.com/mozilla/DeepSpeech.git
cd DeepSpeech/
git reset --hard e00bfd0f413912855eb2312bc1efe3bd2b023b25
```
Note: if you have git-lfs installed, you can disable it for the benchmarks using environment variable GIT_LFS_SKIP_SMUDGE=1.

**Download native libraries**

```bash
OBSOLETE: python util/taskcluster.py --arch gpu --target native_client/ --branch=v0.2.0
```

NOTE: Unfortunately, Mozilla servers do not host native client binaries
for v0.2.0 any more. We provide a mirror of the x86-64 binaries
used in our experiments:

```bash
cd native_client/
wget https://github.com/roscisz/dnn_training_benchmarks/raw/master/TensorFlowV1_DeepSpeech_ldc93s1/native_client.tar.xz
tar xf native_client.tar.xz
cd ..
```

Alternatively, see Dockerfile for instructions how to build the native client libraries.

**Download small dataset**
```bash
python bin/import_ldc93s1.py ldc93s1
```

### Applying the benchmarking patch

```bash
wget https://raw.githubusercontent.com/roscisz/dnn_training_benchmarks/master/TensorFlowV1_DeepSpeech_ldc93s1/deepspeech_benchmarking.patch
git apply deepspeech_benchmarking.patch
```

## Running the benchmarks

In this section we describe the steps to reproduce the [experimental results](#experimental-results),
assuming that the DeepSpeech training program is installed and the benchmarking patch is applied.

### Manually

To run the benchmark, specify the number of "global steps" to be benchmarked using the `benchmark_steps` parameter:

```bash
LD_LIBRARY_PATH=native_client/ CUDA_VISIBLE_DEVICES=0 python ./DeepSpeech.py --train_files=ldc93s1/ldc93s1.csv --dev_files=ldc93s1/ldc93s1.csv --test_files=ldc93s1/ldc93s1.csv --log_level=3 --benchmark_steps=10
```

**Testing batch size on one GPU**

To check the performance for various batch sizes, modify the `train_batch_size` parameter. For example, to use batch size of 128 run:

```bash
LD_LIBRARY_PATH=native_client/ CUDA_VISIBLE_DEVICES=0 python ./DeepSpeech.py --train_files=ldc93s1/ldc93s1.csv --dev_files=ldc93s1/ldc93s1.csv --test_files=ldc93s1/ldc93s1.csv --log_level=3 --benchmark_steps=10 --train_batch_size=128
```
**Testing scalability on many GPUs**

***In-graph replication***

To check the performance of parallel training on multiple GPUs, modify the CUDA_VISIBLE_DEVICES environment variable.
For example, to use GPUs 1 and 2, set CUDA_VISIBLE_DEVICES=1,2 and to use all GPUs in a 4-GPU system, set
CUDA_VISIBLE_DEVICES=0,1,2,3. The in-graph replication method for data-parallel, synchronized training implemented in
Mozilla DeepSpeech will be used.

### run-cluster.sh

The Mozilla DeepSpeech implementation supports distributed training using Distributed TensorFlow with a
parameter server and worker processes. The 'bin/run-cluster.sh' script is helpful for configuring and running
these processes on a single machine. The script accepts an argument in a p:w:g format, where p denotes the 
number of used parameter servers, w denotes the number of worker processes and g denotes the number of
GPUs used by individual worker processes.

For example, running our benchmark on the distributed training
application using 1 parameter server and 4 workers using 1 GPU each would require executing the following command:

```bash
LD_LIBRARY_PATH=native_client/ bin/run-cluster.sh 1:4:1 --script="python DeepSpeech.py" --train_files=ldc93s1/ldc93s1.csv --dev_files=ldc93s1/ldc93s1.csv --test_files=ldc93s1/ldc93s1.csv --train_batch_size=64 --epoch=1000 --benchmark_warmup_steps=10 --benchmark_steps=10 --log_level=3 --noshow_progressbar
```

It should be noted that the distributed training introduces a startup overhead, so increasing the number of
warmup steps can be necessary to collect reliable results.

### Docker

We provide a Dockerfile that allows to build and run the benchmark as a Docker image:


```bash
docker build -t deepspeech .
```

If there is a need to share the image between distributed machines, the repository has to be given
in the image tag, and the image has to be pushed in to a Docker repository:

```bash
docker build -t <repositoryname>/deepspeech .
docker push <repositoryname>/deepspeech
```
 
Now, the benchmark can be executed in a Docker container, so that no dependencies need to be installed
on the host machine: 
 
```bash
docker run deepspeech python ./DeepSpeech.py --train_files=ldc93s1/ldc93s1.csv --dev_files=ldc93s1/ldc93s1.csv --test_files=ldc93s1/ldc93s1.csv --log_level=3 --benchmark_steps=10 --train_batch_size=128
```


### Kubernetes

In order to enqueue the benchmark in a Kubernetes installation
[(for example microk8s)](https://gist.github.com/PiotrowskiD/07a57ad0f21e2b4de78454d02b34865c),
create an adequate deployment file (example provided in ds.yaml) and create the resource:

```bash
kubectl create -f ds.yaml
```

The status of the resulting Pod, its detailed description and logs can be fetch as follows:

```bash
kubectl get pod
kubectl describe pod ds
kubectl logs ds
``` 

Unfortunately Kubernetes doesn't take into account other process using GPUs which leads to conflicts if
somebody else runs their jobs manually. Because CUDA_VISIBLE_DEVICES env variable is used inside the
container, it can only chose from GPUs kubernetes assigns to the container. So if we would like to deploy
our training to GPU number 3 we would have to set GPUs limit to 4 and set CVD to "3". That is if the GPUs
are on a single node. If they would be on different ones we could use
[node labels and selectors](https://kubernetes.io/docs/tasks/configure-pod-container/assign-pods-nodes/).


## Experimental results

### Batch size

As expected, performance increases proportionally to batch size, up to a limit depending on GPU memory capacity:  

![batch_size_v100](https://raw.githubusercontent.com/roscisz/dnn_training_benchmarks/master/TensorFlowV1_DeepSpeech_ldc93s1/img/batch_size_v100.png)
![batch_size_gtx1060](https://raw.githubusercontent.com/roscisz/dnn_training_benchmarks/master/TensorFlowV1_DeepSpeech_ldc93s1/img/batch_size_gtx1060.png)

Although using Distributed TensorFlow with Parameter Servers allows distributed training on multiple nodes, it
should be noted that communication with the Parameter Server introduces significant overhead comparing to the
in-graph replication method:  

![batch_size_v100_distributed](https://raw.githubusercontent.com/roscisz/dnn_training_benchmarks/master/TensorFlowV1_DeepSpeech_ldc93s1/img/batch_size_v100_distributed.png)

### MultiGPU scalability

The following results show performance results on NVIDIA® DGX Station™, depending on the choice utilized GPUs. The
results are marked with ID's of the used GPUs, for example '013' means that CUDA_VISIBLE_DEVICES was set to 0,1,3.

![multigpu_128](https://raw.githubusercontent.com/roscisz/dnn_training_benchmarks/master/TensorFlowV1_DeepSpeech_ldc93s1/img/multigpu_128.png)

Interestingly, in the cases of utilizing two GPUs, it is significant which GPUs are used exactly. For example,
combining GPUs 0 and 1 or 2 and 3 results in worse performance. This is probably connected with interconnects between
the GPUs.

Overhead of the Distributed TensorFlow implementation is also visible in the multi-GPU setup:

![multigpu_128_distributed](https://raw.githubusercontent.com/roscisz/dnn_training_benchmarks/master/TensorFlowV1_DeepSpeech_ldc93s1/img/multigpu_128_distributed.png)

The results show that in the investigated setup it is better to run many processes utilizing single GPUs than 
to run one process utilizing multiple GPUs. 
