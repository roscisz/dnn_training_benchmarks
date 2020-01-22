# PyTorch - DCGAN

This example is based on [this implementation](https://github.com/pytorch/examples/tree/master/dcgan)


## Downloading the dataset

You can download the LSUN dataset by cloning [this repo](https://github.com/fyu/lsun) and running
```
python download.py -c bedroom
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

There is not that much more to do in this case.

```
from torch import distributed, nn


<argument parser>

parser.add_argument('--backend', type=str, default='gloo', help='Name of the backend to use.')
parser.add_argument(
    '-i',
    '--init-method',
    type=str,
    default='tcp://127.0.0.1:23456',
    help='URL specifying how to initialize the package.')
parser.add_argument('-s', '--world-size', type=int, default=1, help='Number of processes participating in the job.')
parser.add_argument('-r', '--rank', type=int, default=0, help='Rank of the current process.')


if opt.world_size > 1:
        distributed.init_process_group(
            backend=opt.backend,
            init_method=opt.init_method,
            world_size=opt.world_size,
            rank=opt.rank,
        )


<code with model creation, in this case Generator is called netG>
netG = nn.parallel.DistributedDataParallel(netG)

<another model creation in this case discriminator called netD>
netD = nn.parallel.DistributedDataParallel(netD)
```

That is basically it.
Although for it to make sense you need to make your sampler distributed too.

```

from torch.utils import data

<sampler creation>
sampler = data.DistributedSampler(dataset)
```


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
