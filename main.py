import argparse
import copy
import os
import random

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import SGD

import numpy as np

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import pytorch_classification.models.cifar as cifar_models
from pytorch_classification import cifar
import torch.utils.data as data

#TODO: LR Decay
#TODO: Controllare che la deep copy funzioni

model_names = sorted(name for name in cifar_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(cifar_models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--block-name', type=str, default='BasicBlock',
                    help='the building block for Resnet and Preresnet: BasicBlock, Bottleneck (default: Basicblock for cifar10/cifar100)')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
# Validate dataset
assert args.dataset == 'cifar10' or args.dataset == 'cifar100', 'Dataset can only be cifar10 or cifar100.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

class NeuronId:
    def __init__(self, param_id, neuron_index):
        self.param_id = param_id
        self.neuron_index = neuron_index

class SplitOptimizer(optim.Optimizer):
    def __init__(self, params, lrs=[1e-3, 2e-3], momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        
        self.learning_rates = lrs
        parameters = copy.copy(list(params))

        neurons = []
        parameter_configurations = {}
        for parameter in parameters:
            for i in range(len(parameter)):
                neuron_id = NeuronId(id(parameter), i)
                neurons.append(neuron_id)

            parameter_configurations[id(parameter)] = [None] * len(parameter)

        neuron_slices = []

        np.random.shuffle(neurons)

        base_slice_size = len(neurons) // len(lrs)

        for _ in range(len(lrs) - 1):
            neuron_slices.append(neurons[:base_slice_size])
            neurons = neurons[base_slice_size:]

        # Add the remaining parameters
        neuron_slices.append(neurons)

        for i, neuron_slice in enumerate(neuron_slices):
            for neuron_id in neuron_slice:
                parameter_configurations[neuron_id.param_id][neuron_id.neuron_index] = i

        self.parameter_configurations = parameter_configurations
        print(parameter_configurations[id(list(parameters)[0])])
        
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
    
        super().__init__(parameters, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    print('No grad!')
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                assert id(p) in self.parameter_configurations.keys()
                parameter_configuration = self.parameter_configurations[id(p)]

                for i in range(len(p)):
                    learning_rate = self.learning_rates[parameter_configuration[i]]
                    p.data[i] += -learning_rate * d_p[i]

        return loss
        

def main():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if args.dataset == 'cifar10':
        dataloader = datasets.CIFAR10
        num_classes = 10
    else:
        dataloader = datasets.CIFAR100
        num_classes = 100


    trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = dataloader(root='./data', train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    if args.arch.startswith('resnext'):
        model = cifar_models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = cifar_models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = cifar_models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = cifar_models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    block_name=args.block_name,
                )
    else:
        model = cifar_models.__dict__[args.arch](num_classes=num_classes)

    learning_rates = [5e-4, 1e-3, 2e-3, 4e-3]

    models = [copy.deepcopy(model) for _ in range(len(learning_rates) + 1)]

    start_epoch = 0

    criterion = nn.CrossEntropyLoss()

    optimizers = [
        SplitOptimizer(models[0].parameters(), lrs=learning_rates, momentum=args.momentum, weight_decay=args.weight_decay)
    ]

    for i, learning_rate in enumerate(learning_rates):
        optimizer = optim.SGD(models[i + 1].parameters(), lr=learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizers.append(optimizer)

    print('CUDA: {}'.format(use_cuda))

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        # adjust_learning_rate(optimizer, epoch)

        #print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        for i, (model, optimizer) in enumerate(zip(models, optimizers)):
            train_loss, train_acc = cifar.train(trainloader, model, i, criterion, optimizer, epoch, use_cuda)
            test_loss, test_acc = cifar.test(testloader, model, i, criterion, epoch, use_cuda)

            name = type(model).__name__
            print('Test accuracy for model {} ({}): {}'.format(i + 1, name, test_acc))
        print('==========')

def adjust_learning_rate(optimizer, epoch):
    if isinstance(optimizer, SGD):
        if epoch in args.schedule:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * args.gamma
    elif isinstance(optimizer, SplitOptimizer):
        optimizer.learning_rates = [lr * args.gamma for lr in optimizer.learning_rates]
    else:
        assert False

if __name__ == '__main__':
    main()