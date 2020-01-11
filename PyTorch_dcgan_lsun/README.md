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
usage: main.py [-h] --dataset DATASET --dataroot DATAROOT [--workers WORKERS]
               [--batchSize BATCHSIZE] [--imageSize IMAGESIZE] [--nz NZ]
               [--ngf NGF] [--ndf NDF] [--niter NITER] [--lr LR]
               [--beta1 BETA1] [--cuda] [--ngpu NGPU] [--netG NETG]
               [--netD NETD]

distributed arguments:
  --init-method metod,addr, and port to communicate between the nodes [tcp://127.0.0.1:20010]
  --rank current process id
  --world-size number of processes 

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     cifar10 | lsun | mnist |imagenet | folder | lfw | fake
  --dataroot DATAROOT   path to dataset
  --workers WORKERS     number of data loading workers
  --batchSize BATCHSIZE input batch size
  --imageSize IMAGESIZE the height / width of the input image to network
  --nz NZ               size of the latent z vector
  --ngf NGF
  --ndf NDF
  --niter NITER         number of epochs to train for
  --lr LR               learning rate, default=0.0002
  --beta1 BETA1         beta1 for adam. default=0.5
  --cuda                enables cuda
  --ngpu NGPU           number of GPUs to use
  --netG NETG           path to netG (to continue training)
  --netD NETD           path to netD (to continue training)
  --outf OUTF           folder to output images and model checkpoints
  --manualSeed SEED     manual seed
  --classes CLASSES     comma separated list of classes for the lsun data set


```
