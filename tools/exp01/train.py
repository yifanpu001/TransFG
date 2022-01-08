# coding=utf-8
from __future__ import absolute_import, division, print_function

import hf_env
hf_env.set_env('202105')
import pickle
from ffrecord.torch import Dataset, DataLoader

import os
import math
import time
import random
import logging
import argparse
import numpy as np
from enum import Enum

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter

from models.modeling import VisionTransformer, CONFIGS
from utils.data_utils import get_loader

logger = logging.getLogger(__name__)
best_acc1, best_epoch = 0.0, 0


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


def adjust_learning_rate(optimizer, init_lr, epoch_total, warmup_epochs, epoch_cur, num_iter_per_epoch, i_iter):
    """
    cosine learning rate with warm-up
    """
    if epoch_cur < warmup_epochs:
        # T_cur = 1, 2, 3, ..., (T_total - 1)
        T_cur = 1 + epoch_cur * num_iter_per_epoch + i_iter
        T_total = 1 + warmup_epochs * num_iter_per_epoch
        lr = (T_cur / T_total) * init_lr
    else:
        # T_cur = 0, 1, 2, 3, ..., (T_total - 1)
        T_cur = (epoch_cur - warmup_epochs) * num_iter_per_epoch + i_iter
        T_total = (epoch_total - warmup_epochs) * num_iter_per_epoch
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * T_cur / T_total))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(state, os.path.join(save_dir, filename))
    if is_best:
        torch.save(state, os.path.join(save_dir, 'model_best.pth.tar'))


def estimated_time(t_start, cur_epoch, start_epoch, total_epoch):
    t_curr = time.time()
    eta_total = (t_curr - t_start) / (cur_epoch + 1 - start_epoch) * (total_epoch - cur_epoch - 1)
    eta_hour = int(eta_total // 3600)
    eta_min = int((eta_total - eta_hour * 3600) // 60)
    eta_sec = int(eta_total - eta_hour * 3600 - eta_min * 60)
    # args.print_custom(f'[INFO] Finished epoch:{epoch:02d};  ETA {eta_hour:02d} h {eta_min:02d} m {eta_sec:02d} s')
    return f'Finished iter:{cur_epoch:05d}/{total_epoch:05d};  ETA {eta_hour:02d} h {eta_min:02d} m {eta_sec:02d} s'


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


""" part of main """
def get_args():
    parser = argparse.ArgumentParser()
    # Model Related
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="/opt/tiger/minist/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.   'logs/pretrained_ViT/imagenet21k_ViT-B_16.npz' for HF")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="load pretrained model")
    # Data Related
    parser.add_argument("--dataset", choices=["CUB_200_2011", "car", "dog", "nabirds", "INat2017"], default="CUB_200_2011",
                        help="Which dataset.")
    parser.add_argument('--data_root', type=str, default='/root/data/public/',
                    help='/ceph-jd/pub/jupyter/pxr/notebooks/pyf/datasets for HF')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument("--img_size", default=448, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    # Directory Related
    parser.add_argument("--output_dir_root", default="/root/share/TransFG/output", type=str,
                        help="output_dir's root")
    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where checkpoints will be written.")
    # Optimizer & Learning Schedule
    parser.add_argument("--lr", "--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate.")
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument("--warmup_epochs", default=5, type=int,
                        help="Step of training to perform learning rate warmup for.")
    # For a Specific Experiment
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--round', type=int, help="repeat same hyperparameter round")
    # Experiment Hyper-parameters
    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")
    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")
    ## ISDA
    parser.add_argument('--lambda_0', type=float, required=True,
                    help='The hyper-parameter \lambda_0 for ISDA, select from {1, 2.5, 5, 7.5, 10}. '
                         'We adopt 1 for DenseNets and 7.5 for ResNets and ResNeXts, except for using 5 for ResNet-101.')

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
        f'{args.dataset}_{args.model_type}_bs{args.train_batch_size}_lr{args.lr}_wd{args.weight_decay}_epochs{args.epochs}_wmsteps{args.warmup_epochs}_{args.split}_lbd{args.lambda_0:06.2f}_round{args.round}/'
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set seed
    set_seed(args)

    # Start Multiprocessing
    mp.spawn(main_worker, nprocs=torch.cuda.device_count(), args=(torch.cuda.device_count(), args))  # ngpus_per_node = torch.cuda.device_count() == 8


def main_worker(local_rank, ngpus_per_node, args):
    """
    for multi-nodes training, the gpu is still 0-7, but each node has 0-7
    """
    global best_acc1, best_epoch
    args.local_rank = local_rank


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
    args.ngpus_per_node = ngpus_per_node                # 每台机器的GPU个数 (8 for huanfnag)
    args.world_size = hosts * args.ngpus_per_node
    args.world_rank = rank * args.ngpus_per_node + args.local_rank
    dist.init_process_group(backend='nccl', init_method=f'tcp://{ip}:{port}', world_size=args.world_size, rank=args.world_rank)
    args.is_main_proc = (args.world_rank == 0)


    # Model & Tokenizer Setup
    args, model = setup_model(args)


    # DistributedDataParallel
    args.train_batch_size = int(args.train_batch_size / args.ngpus_per_node)
    args.workers = int((args.workers + args.ngpus_per_node - 1) / args.ngpus_per_node)
    torch.cuda.set_device(args.local_rank)
    model.cuda(args.local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])


    # Prepare optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)


    cudnn.benchmark = True


    # Prepare dataset
    train_loader, test_loader, train_sampler = get_loader(args)


    # Init scores_all.csv
    if args.is_main_proc:
        if not os.path.exists(args.output_dir + '/scores_all.csv'):
            with open(args.output_dir + '/scores_all.csv', "a") as f:
                f.write(f'epoch, lr, loss_train, acc1_train, loss_test, acc1_test, acc1_test_best,\n')


    # Auto Resume
    resume_dir = os.path.join(args.output_dir, "save_models", "checkpoint.pth.tar")
    if os.path.exists(resume_dir):
        logger.info(f'[INFO] resume dir: {resume_dir}')
        ckpt = torch.load(resume_dir, map_location='cpu')
        args.start_epoch = ckpt['epoch']
        model.module.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        curr_acc1 = ckpt['curr_acc1']
        best_acc1 = ckpt['best_acc1']
        logger.info(f'[INFO] Auto Resume from {resume_dir}, from  finished epoch {args.start_epoch}, with acc_best{best_acc1}, acc_curr {curr_acc1}.')


    # Start Train
    logger.info("***** Running training *****")

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        loss_train, acc1_train = train(train_loader, model, optimizer, epoch, args)
        loss_test, acc1_test = validate(test_loader, model, args)

        if args.is_main_proc:
            writer.add_scalar("train/loss", scalar_value=loss_train, global_step=epoch)
            writer.add_scalar("train/acc1", scalar_value=acc1_train, global_step=epoch)
            writer.add_scalar("train/lr", scalar_value=get_lr(optimizer), global_step=epoch)
            writer.add_scalar("test/loss", scalar_value=loss_test, global_step=epoch)
            writer.add_scalar("test/acc1", scalar_value=acc1_test, global_step=epoch)

            is_best = acc1_test > best_acc1
            best_acc1 = max(acc1_test, best_acc1)
            if is_best:
                best_epoch = epoch

            with open(args.output_dir + '/scores_all.csv', "a") as f:
                f.write(
                    f"{epoch:3d}, {get_lr(optimizer):15.12f}, {loss_train:9.8f}, {acc1_train:6.3f}, {loss_test:9.8f}, {acc1_test:6.3f}, {best_acc1:6.3f},\n"
                )

            save_checkpoint(
                {'epoch': epoch + 1,
                 'state_dict': model.module.state_dict(),
                 'optimizer' : optimizer.state_dict(),
                 'curr_acc1': acc1_test,
                 'best_acc1': best_acc1,
                 'best_epoch': best_epoch,
                }, is_best, save_dir=os.path.join(args.output_dir, 'save_models')
            )

            logger.info(estimated_time(start_time, epoch, args.start_epoch, args.epochs))

    writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc1)
    logger.info("Last Accuracy: \t%f" % acc1_test)
    logger.info("Training Complete.")


def train(train_loader, model, optimizer, epoch, args):
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
        images = images.cuda(args.local_rank, non_blocking=True)
        target = target.cuda(args.local_rank, non_blocking=True)
        data_time.update(time.time() - end)

        # adjust learning rate
        adjust_learning_rate(optimizer, init_lr=args.lr,
                             epoch_total=args.epochs, warmup_epochs=args.warmup_epochs, epoch_cur=epoch,
                             num_iter_per_epoch=len(train_loader), i_iter=i)

        # compute output
        loss, logits = model(images, target, args.lambda_0 * (epoch / args.epochs))
        loss = loss.mean()
        preds = logits

        # measure accuracy and record loss
        acc1, acc5 = accuracy(preds, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % args.print_freq == 0) and args.is_main_proc:
            progress.display(i)
    
    return losses.avg, top1.avg


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
            images = images.cuda(args.local_rank, non_blocking=True)
            target = target.cuda(args.local_rank, non_blocking=True)

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
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i % args.print_freq == 0) and args.is_main_proc:
                progress.display(i)
        
        if args.is_main_proc:
            progress.display_summary()

    return losses.avg, top1.avg


if __name__ == '__main__':
    main()