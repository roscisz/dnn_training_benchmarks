# PyTorch - DCGAN

This example is based on [this implementation](https://github.com/pytorch/examples/tree/master/dcgan)
```
git clone https://github.com/pytorch/examples.git
```

## Downloading the dataset

You can download the LSUN dataset by cloning [this repo](https://github.com/fyu/lsun) 

```
git cllone https://github.com/fyu/lsun.git
```
and running
```
python download.py -c bedroom
```

For testing and/or benchmarking purposes it is also possible to the train the model on fake dataset. To do this run this command instead of the one given in the Usage section:

```
CUDA_VISIBLE_DEVICES=0 python main.py --init-method tcp://127.0.0.1:20011 --rank 0 --world 2 --dataset fake --cuda
CUDA_VISIBLE_DEVICES=1 python main.py --init-method tcp://127.0.0.1:20011 --rank 1 --world 2 --dataset fake --cuda
```

## Making the code parallel

It is actually very easy to allow PyTorch code to run on multiple GPUs

### Locally

All you need to do to make a data parallel model that can be run on multiple GPUs locally:
```
from torch import nn
<your code including creating the model>
model = nn.DataParallel(model)
```
That's it. After that you just control which GPUs you would like your code to run on by specifing their number in CUDA_VISIBLE_DEVICES before your command (tensorhive takes care of that for you when you specify the devices while creating the task).

### Distributed

To make the code distributed apply provided patch by running:

```
git apply distributed.patch
```

You might want to checkout commit 0c1654d6913f77f09c0505fb284d977d89c17c1a to make sure there are no conflicts before applying the patch.

To learn how to easily adapt PyTorch code for distributed learning please examine the patch. 

## Usage
```
# set accroding to your configuration
export GLOO_SOCKET_IFNAME=eth0 
```

```
CUDA_VISIBLE_DEVICES=0 python main.py --init-method tcp://127.0.0.1:20011 --rank 0 --world 2 --dataset lsun --dataroot <path to lsun dataset> --cuda
CUDA_VISIBLE_DEVICES=1 python main.py --init-method tcp://127.0.0.1:20011 --rank 1 --world 2 --dataset lsun --dataroot <path to lsun dataset> --cuda

For more details about the usage refer to the [original repo](https://github.com/pytorch/examples/tree/master/dcgan)
```