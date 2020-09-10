#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:50:36 2020

@author: fabian
"""


import argparse
import os
import time
import random
import sys
import numpy as np


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


metadata_default = '/scratch1/msc20f10/data/classification/metadata.mat'
parser = argparse.ArgumentParser(description='Touch-Classification.')
parser.add_argument('--dataset', default=metadata_default,
                    help="Path to metadata.mat file.")
parser.add_argument('--nframes', type=int, default=1,
                    help='Number of input frames [1--8]')
parser.add_argument('--checkpointDir', default='checkpoints',
                    help="Where to store checkpoints during training.")
parser.add_argument('--clusterTrain', type=str2bool, default=False,
                    help="Use clustering for the training data or not.")
parser.add_argument('--mask', type=str2bool, default=False,
                    help="Reduce the input to only valid sensels.")
parser.add_argument('--gpu', type=int, default=None,
                    help="ID number of the GPU to use [0--4]. \
                        If left unspecified, keras will all visible GPUs.")
parser.add_argument('--dropout', type=float, default=0.2,
                    help="Dropout to be applied.")
parser.add_argument('--test', type=str2bool, nargs='?', const=True,
                    default=False, help="Tests a previously trained model.")
args = parser.parse_args()

# This line makes only the chosen GPU visible.
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from datatools.ObjectClusterDataset import ObjectClusterDataset
from tactile_nn_model import TCN


torch.manual_seed(13751)

epochs = 90
batch_size = 32
nFrames = args.nframes
# metaFile = '../../Research/STAG_MIT/classification_lite/metadata.mat'
metaFile = args.dataset
checkpointDir = args.checkpointDir
input_channels = 1
seq_length = int(32*32 / input_channels) if not args.mask else 548
steps = 0
clip = -1
log_interval = 30
num_classes = 27

saveDir = '/home/msc20f10/Python_Code/results/\
    tcn/{}_frames_'.format(nFrames)

# Mask that only passes the physically present sensor points
mask = np.array([np.ones(32), np.ones(32), np.ones(32),
                 np.concatenate((np.zeros(14), np.ones(18))),
                 np.concatenate((np.zeros(14), np.ones(18))),
                 np.concatenate((np.zeros(14), np.ones(18))),
                 np.ones(32), np.ones(32), np.ones(32),
                 np.concatenate((np.zeros(14), np.ones(18))),
                 np.ones(32), np.ones(32), np.ones(32),
                 np.concatenate((np.zeros(14), np.ones(18))),
                 np.concatenate((np.zeros(14), np.ones(18))),
                 np.ones(32), np.ones(32), np.ones(32),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3))),
                 np.concatenate((np.zeros(25), np.ones(4), np.zeros(3)))])
mask = mask.astype(np.bool)


def loadDatasets(split='train', shuffle=True, useClusterSampling=False):
        return torch.utils.data.DataLoader(
            ObjectClusterDataset(split=split,
                                 doAugment=(split == 'train'),
                                 doFilter=True,
                                 sequenceLength=nFrames,
                                 metaFile=metaFile,
                                 useClusters=useClusterSampling),
            batch_size=batch_size, shuffle=shuffle, num_workers=0)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.data.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.data.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res[0], res[1]


def train(epoch, model, train_loader):
    """"Trains the model for one epoch"""
    global steps
    training_loss = 0
    train_loss = 0
    correct = 0
    (prec1_sum, prec3_sum) = (0, 0)
    model.train()
    train_loader.dataset.refresh()
    for batch_idx, (row, image, pressure, objectId) in enumerate(train_loader):
        # For now just take the average of all input frames
        if nFrames > 1:
            pressure = torch.sum(pressure, dim=1)/nFrames
        if args.mask:
            pressure = pressure[:,:,mask]
        objectId = objectId.squeeze()
        pressure, objectId = pressure.cuda(), objectId.cuda()
        # Resizes the input from 2D to 1D
        pressure = pressure.view(-1, input_channels, seq_length)

        # Zero gradients before accumulating them
        optimizer.zero_grad()
        output = model(pressure)
        # Calculate loss
        loss = criterion(output, objectId)
        # Propagate the loss backward
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        # Change weights
        optimizer.step()
        train_loss += loss
        training_loss += loss
        steps += seq_length
        # Using the MIT accuracy function
        (prec1, prec3) = accuracy(output, objectId,
                                  topk=(1, min(3, num_classes)))
        prec1_sum += prec1
        prec3_sum += prec3
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(objectId.data.view_as(pred)).sum()

        # Print results every log_interval batch
        if batch_idx > 0 and batch_idx % log_interval == 0:
            sys.stdout.write('\033[K')
            sys.stdout.flush()
            print(('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                  + '\tAccuracy: {}/{} ({:.0f}%)\tSteps: {}').format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                train_loss.item()/log_interval,
                correct, len(train_loader.dataset),
                100. * correct / len(train_loader.dataset),
                steps), end="\r", flush=True)
            train_loss = 0

    print('')
    training_loss = training_loss//len(train_loader)
    prec1_sum = prec1_sum/len(train_loader)
    prec3_sum = prec3_sum/len(train_loader)
    return training_loss, prec1_sum, prec3_sum


def test(model, test_loader):
    """Evaluates the models performance"""
    model.eval()
    test_loss = 0
    with torch.no_grad():
        correct = 0
        (prec1_sum, prec3_sum) = (0, 0)
        for row, image, pressure, objectId in test_loader:
            if nFrames > 1:
                pressure = torch.sum(pressure, dim=1)/nFrames
            if args.mask:
                pressure = pressure[:,:,mask]
            objectId = objectId.squeeze()
            pressure, objectId = pressure.cuda(), objectId.cuda()
            pressure = pressure.view(-1, input_channels, seq_length)

            output = model(pressure)
            test_loss += criterion(output, objectId).item()
            # Using the MIT accuracy function
            (prec1, prec3) = accuracy(output, objectId,
                                      topk=(1, min(3, num_classes)))
            prec1_sum += prec1
            prec3_sum += prec3
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(objectId.data.view_as(pred)).sum()

        test_loss /= len(test_loader.dataset)
        print(('Test: Loss: {:.4f}\tAccuracy:\tTop 1: {}/{} ({:.1f}%)'
               + '\tTop 3: ({:.1f}%)\n').format(
            test_loss, correct, len(test_loader.dataset),
            prec1_sum/len(test_loader), prec3_sum/len(test_loader)))

        return test_loss


if __name__ == "__main__":
    # The receptive field has size: 1+2*(kernel_size-1)*(2^(layers)-1)
    model = TCN(input_size=input_channels,
                output_size=num_classes,
                num_channels=[32, 32, 64, 64, 64, 64, 64, 128],
                kernel_size=7,
                dropout=args.dropout)
    model.cuda()
    
    train_loader = loadDatasets('train', shuffle=True,
                            useClusterSampling=args.clusterTrain)
    test_loader = loadDatasets('test', shuffle=False,
                               useClusterSampling=False)
    # testcluster_loader = loadDatasets('test', shuffle=False,
    #                                   useClusterSampling=True)
    
    criterion = nn.NLLLoss()
    base_lr = 1e-3
    optimizer = getattr(optim, 'Adam')(model.parameters(), lr=base_lr)
    
    train_loss = []
    test_loss = []
    # testcluster_loss = []
    period = 20
    print('Initial learning rate = {:f}'.format(base_lr))
    for epoch in range(1, epochs+1):
        train_loss.append(train(epoch, model, train_loader))
        test_loss.append(test(model, test_loader))
        # testcluster_loss.append(test(model, testcluster_loader))
        gamma = 0.1 ** (1.0/period)
        lr_default = base_lr * (gamma ** (epoch))
        print('New lr_default = {:f}'.format(lr_default))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_default

    # np.save(saveDir+'train', np.array(train_loss),
    #         allow_pickle=True, fix_imports=True)
    # np.save(saveDir+'test', np.array(test_loss),
    #         allow_pickle=True, fix_imports=True)
