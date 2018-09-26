import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
import utils
from utils import LRScheduler, ColorAugmentation
from tensorboardX import SummaryWriter

import yaml
import subprocess
import imp
import numpy as np

from six import text_type as _text_type
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',default='data',
                    help='path to dataset')
parser.add_argument('--cuda', action='store_true', help='use cuda?',default=True)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--config', default='configs/resnet_v2_152.yaml')


best_prec1 = 0
from importlib import import_module



def main():
    global args, best_prec1
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)






    proc = subprocess.Popen( 'mmdownload -f '+args.f+' -n '+args.n+' -o ./models/'+args.f+'_'+args.n+'/' , shell=True, executable='/bin/bash')
    proc.communicate()
    proc = subprocess.Popen( 'mmconvert -sf '+args.srcFramework+' --inputShape 224,224,3 -in models/'+args.f+'_'+args.n+'/imagenet_'+args.n+'.ckpt.meta -iw models/'+args.f+'_'+args.n+'/imagenet_'+args.n+'.ckpt --dstNodeName MMdnn_Output -df '+args.dstFramework+' -om models/'+args.n+'.pth' , shell=True, executable='/bin/bash')
    proc.communicate()

    MainModel = imp.load_source('MainModel', "models/"+args.model+".py")
    model = torch.load("models/"+args.n+".pth")



    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        model=model.cpu()
        criterion = nn.CrossEntropyLoss().cpu()


    # define loss function (criterion) and optimizer


    optimizer = torch.optim.SGD(model.parameters(), args.base_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # auto resume from a checkpoint
    model_dir = args.model_dir
    start_epoch = 0
    if not os.path.exists(model_dir) :
        os.makedirs(model_dir)
    if args.evaluate:
        pass


    else:
        best_prec1, start_epoch = utils.load_state(model_dir, model, optimizer=optimizer)
    writer = SummaryWriter(model_dir)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
      datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
      ])),
      batch_size=args.batch_size, shuffle=False,
      num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, 0, writer)
        return

    train_dataset_multi_scale = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        #    ColorAugmentation(),
            normalize,
        ]))

    train_dataset = datasets.ImageFolder(
      traindir,
      transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #ColorAugmentation(),
        normalize,
      ]))


    train_sampler = None

    train_loader_multi_scale = torch.utils.data.DataLoader(
        train_dataset_multi_scale, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
      num_workers=args.workers, pin_memory=True, sampler=train_sampler)




    niters = len(train_loader)

    lr_scheduler = LRScheduler(optimizer, niters, args)

    for epoch in range(start_epoch, args.epochs):


        # train for one epoch
        if epoch < args.epochs - 5:
            train(train_loader_multi_scale, model, criterion, optimizer, lr_scheduler, epoch, writer)
        else:
            train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, writer)




        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, writer)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        utils.save_checkpoint(model_dir, {
          'epoch': epoch + 1,
          'model': args.model,
          'state_dict': model.state_dict(),
          'best_prec1': best_prec1,
          'optimizer': optimizer.state_dict(),
        }, is_best)

def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        lr_scheduler.update(i, epoch)
        target = target.cuda(non_blocking=True)
        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            niter = epoch * len(train_loader) + i
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], niter)
            writer.add_scalar('Train/Avg_Loss', losses.avg, niter)
            writer.add_scalar('Train/Avg_Top1', top1.avg / 100.0, niter)
            writer.add_scalar('Train/Avg_Top5', top5.avg / 100.0, niter)



def validate(val_loader, model, criterion, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model=model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        niter = (epoch + 1)
        writer.add_scalar('Eval/Avg_Loss', losses.avg, niter)
        writer.add_scalar('Eval/Avg_Top1', top1.avg / 100.0, niter)
        writer.add_scalar('Eval/Avg_Top5', top5.avg / 100.0, niter)

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()
