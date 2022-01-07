# coding=utf-8
from __future__ import absolute_import, division, print_function

import hf_env
hf_env.set_env('202105')
import pickle
from ffrecord.torch import Dataset, DataLoader


import logging
import argparse
import os
import random
import numpy as np
import time
from enum import Enum

from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP

from models.modeling_pure import VisionTransformer, CONFIGS
from utils.data_utils import get_loader
from utils.dist_util import get_world_size
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule

logger = logging.getLogger(__name__)


""" some tools """
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        logger.info('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))
        logger.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

""" part of main """
def get_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    # parser.add_argument("--name", required=True,
    #                     help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds", "INat2017"], default="CUB_200_2011",
                        help="Which dataset.")
    parser.add_argument('--data_root', type=str, default='/root/data/public/',
                    help='/ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets for HF')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="/opt/tiger/minist/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.   'logs/pretrained_ViT/imagenet21k_ViT-B_16.npz' for HF")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="load pretrained model")
    parser.add_argument("--output_dir_root", default="/root/share/TransFG/output", type=str,
                        help="output_dir's root")
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument("--warmup_epochs", default=5, type=int,
                        help="Step of training to perform learning rate warmup for.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")

    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")

    parser.add_argument('--round', type=int, help="repeat same hyperparameter round")

    # not important
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
    args = parser.parse_args()
    return args


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def setup_model(args):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.split = args.split
    config.slide_step = args.slide_step

    if args.dataset == "CUB_200_2011":
        num_classes = 200
    elif args.dataset == "car":
        num_classes = 196
    elif args.dataset == "nabirds":
        num_classes = 555
    elif args.dataset == "dog":
        num_classes = 120
    elif args.dataset == "INat2017":
        num_classes = 5089

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, smoothing_value=args.smoothing_value)

    model.load_from(np.load(os.path.join(args.output_dir_root, args.pretrained_dir)))

    # Load pretrained model
    if args.pretrained_model is not None:
        pretrained_model = torch.load(args.pretrained_model)['model']
        model.load_state_dict(pretrained_model)

    return args, model


def main():
    # Get args
    args = get_args()

    # Setup data_root
    args.data_root = '{}/{}'.format(args.data_root, args.dataset)

    # Setup save path
    args.output_dir = os.path.join(
        args.output_dir_root,
        args.output_dir,
        f'{args.dataset}_{args.model_type}_bs{args.train_batch_size}_lr{args.learning_rate}_wd{args.weight_decay}_epochs{args.epochs}_wmsteps{args.warmup_steps}_{args.split}_round{args.round}/'
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set seed
    set_seed(args)

    # Start Multiprocessing
    mp.spawn(main_worker, nprocs=torch.cuda.device_count(), args=(torch.cuda.device_count(), args))  # ngpus_per_node = torch.cuda.device_count() == 8


def main_worker(gpu, ngpus_per_node, args):
    """
    for multi-nodes training, the gpu is still 0-7, but each node has 0-7
    """
    global best_prec1, best_epoch
    args.gpu = gpu
    local_rank = gpu

    # Setup logging & TensorBoard Writer
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
                        filename=os.path.join(args.output_dir, 'screen_output.log'))
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboardlog"))

    # Multiprocessing
    ip = os.environ['MASTER_IP']                        # same for each gpu, each node
    port = os.environ['MASTER_PORT']                    # same for each gpu, each node
    hosts = int(os.environ['WORLD_SIZE'])               # 机器个数, 每台机器有8张卡   相当于脚本里的--nodes=4的数量
    rank = int(os.environ['RANK'])                      # 当前机器编号, 用了四台机器的话，编号分别是 0,1,2,3
    gpus = torch.cuda.device_count()                    # 每台机器的GPU个数 (8)
    args.ngpus_per_node = torch.cuda.device_count()          # 每台机器的GPU个数
    args.is_main_proc = (rank * gpus + local_rank == 0)
    args.world_size = hosts * gpus
    dist.init_process_group(backend='nccl', init_method=f'tcp://{ip}:{port}', world_size=hosts*gpus, rank=rank*gpus+local_rank)

    # Model & Tokenizer Setup
    args, model = setup_model(args)

    # DistributedDataParallel
    args.train_batch_size = int(args.train_batch_size / args.ngpus_per_node)
    args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # Prepare optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    # Prepare scheduler
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.epochs)

    cudnn.benchmark = True

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Start Train
    logger.info("***** Running training *****")
    
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for epoch in range(args.start_epoch, args.epochs):
        # train_sampler?
        train(train_loader, model, scheduler, optimizer, epoch, args)
        acc1 = validate(test_loader, model, args)
        print(acc1)


def train(train_loader, model, scheduler, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        data_time.update(time.time() - end)

        # compute output
        loss, logits = model(images, target)
        loss = loss.mean()
        preds = logits

        # measure accuracy and record loss
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % args.print_freq == 0) and args.is_main_proc:
            progress.display(i)


def validate(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            loss, logits = model(images, target)
            loss = loss.mean()
            preds = logits

            # measure accuracy and record loss
            acc1, acc5 = accuracy(preds, target, topk=(1, 5))

            dist.all_reduce(acc1)
            acc1 /= args.world_size
            dist.all_reduce(acc5)
            acc5 /= args.world_size
            dist.all_reduce(loss)
            loss /= args.world_size

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i % args.print_freq == 0) and args.is_main_proc:
                progress.display(i)
        
        if args.is_main_proc:
            progress.display_summary()

    return top1.avg


if __name__ == '__main__':
    main()