# PyTorch - DCGAN

This example is based on [the PyTorch DCGAN example](https://github.com/pytorch/examples/tree/master/dcgan).
```
git clone https://github.com/pytorch/examples.git
cd examples/dcgan/
git checkout 0c1654d6913f77f09c0505fb284d977d89c17c1a
pip install -r requirements.txt
```

## Downloading the dataset

You can download the LSUN dataset by cloning [the LSUN Demo Code repo](https://github.com/fyu/lsun):

```
git clone https://github.com/fyu/lsun.git
```
and running
```
cd lsun/
python download.py -c bedroom
cd ..
```

For testing and/or benchmarking purposes it is also possible to the train the model on fake dataset. To do this, set the parameter `--dataset fake` when running the training program (ommit --dataroot parameter).

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

To make the code support distributed training, copy and apply the provided patch:

```
git apply distributed.patch
```


To learn how to easily adapt PyTorch code for distributed learning please examine the patch. 

## Usage
```
# set according to your configuration
export GLOO_SOCKET_IFNAME=eth0 
```

```
# Run the following commands (in the dcgan/ directory) simultaneously, on the appropriate hosts:
CUDA_VISIBLE_DEVICES=0 python main.py --init-method tcp://127.0.0.1:20011 --rank 0 --world-size 2 --dataset lsun --dataroot <path to lsun dataset> --cuda
CUDA_VISIBLE_DEVICES=1 python main.py --init-method tcp://127.0.0.1:20011 --rank 1 --world-size 2 --dataset lsun --dataroot <path to lsun dataset> --cuda
```
For more details about the usage refer to the [original repo](https://github.com/pytorch/examples/tree/master/dcgan)
