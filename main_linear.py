from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
#import tensorboard_logger as tb_logger
from torchvision import transforms, datasets

import numpy

from tqdm import tqdm

#from main_ce import set_loader
from dataloader import MyDataSet, ExpDataSet
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
#from networks.resnet_big import SupConResNet, LinearClassifier
from networks.xception import SupConXception, LinearClassifier

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=500,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='xception')
    parser.add_argument('--dataset', type=str, default='celeb',
                        choices=['cifar10', 'cifar100', 'path', 'celeb', 'faceforensics'], help='dataset')
    parser.add_argument('--test_method', type=str, default='DF',choices=['DF', 'F2F', 'FShifter', 'FSwap', 'NT'])

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default= \
            #'save/SupCon/path_linear_models/path_resnet50_lr_0.1_decay_0_bsz_128_180/ckpt_epoch_60.pth', \
            'save/SupCon/faceforensics_models/test/FShifter/SupCon_faceforensics_xception_lr_0.05_decay_0.0001_bsz_128_temp_0.1_trial_SupCon_FSt/ckpt_epoch_85.pth',\
            #'save/SupCon/celeb_models/abla/SupCon_celeb_xception_lr_0.01_decay_0.0001_bsz_128_temp_0.1_trial_final_1_cosine/ckpt_epoch_100.pth',\
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_linear_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_linear_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}_trial_FSt_85'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
    
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
        
    else:
        opt.n_cls = 2

    return opt

def set_loader(opt, train_txt = 'datasets/FaceForensics/train_txt.txt', test_txt = 'datasets/FaceForensics/test_txt.txt', PATH = 'C:/Users/admin/Desktop/Celeb-DF-v2-face/'):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = (0.5359, 0.4633, 0.4231) 
        std = (0.2446, 0.2361, 0.2358)
    else:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        #transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    elif opt.dataset == 'path':
        train_dataset = MyDataSet('../train_label.txt', transform = train_transform)
        #print(dataset)
        #train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(dataset.__len__()* 0.8), int(dataset.__len__() - int(dataset.__len__() * 0.8))]) 
        val_dataset = MyDataSet('../val_label.txt', transform = val_transform)
    elif opt.dataset == 'celeb':
        train_dataset = MyDataSet(train_txt, PATH = PATH, transform = train_transform)
        #train_dataset = MyDataSet('datasets/Celeb-DF-v2/temp.txt', transform = train_transform)
        val_dataset = MyDataSet(test_txt, PATH = PATH, transform = val_transform)
        #val_dataset = MyDataSet('datasets/Celeb-DF-v2/celeb_real.txt', transform = val_transform)
        #val_dataset = MyDataSet('datasets/Celeb-DF-v2/celeb_synthesis.txt', transform = val_transform)
        #val_dataset = MyDataSet('datasets/Celeb-DF-v2/youtube_real.txt', transform = val_transform)

    elif opt.dataset == 'laohuang':
        train_dataset = MyDataSet('datasets/laohuang/temp.txt',transform = train_transform)
        val_dataset = MyDataSet('datasets/laohuang/pyq_0.txt',transform = val_transform)
    elif opt.dataset == 'faceforensics':
        train_dataset = MyDataSet('datasets/FaceForensics/train{}.txt'.format(opt.test_method), PATH = '../../../data/FaceForensics++_face/', transform = train_transform)
        val_dataset = MyDataSet('datasets/FaceForensics/val{}.txt'.format(opt.test_method), PATH= '../../../data/FaceForensics++_face/', transform= val_transform)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader, val_loader

def set_model(opt):
    model = SupConXception()
    criterion = torch.nn.CrossEntropyLoss()
    
    #print(opt.model)

    classifier = LinearClassifier(num_classes=opt.n_cls)
    
    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        #new_state_dict = {}
        #for k, v in state_dict.items():
        #    k = k.replace("encoder.", "encoder.module.")
        #    new_state_dict[k] = v
        #state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)
        
        #class_ckpt = torch.load('save/SupCon/path_linear_models/path_resnet50_lr_0.01_decay_0_bsz_128_220/ckpt_epoch_35.pth')
        #class_state_dict = class_ckpt['model']
        #classifier.load_state_dict(class_state_dict)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    with tqdm(range(train_loader.__len__())) as p_bar:
        for idx, (images, labels) in enumerate(train_loader):
            data_time.update(time.time() - end)

            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            # warm-up learning rate
            warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

            # compute loss
            with torch.no_grad():
                features = model.encoder(images)
            output = classifier(features.detach())
            #print(output)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            
            #acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            acc1 = accuracy(output, labels, topk=(1))
            #top1.update(acc1[0], bsz)
            top1.update(acc1[0],bsz)

            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            p_bar.update(1)
            p_bar.set_description('train: epoch {} [{}]'.format(epoch, idx))
            p_bar.set_postfix({"loss": losses.avg, "acc": top1.avg[0].item()})
            # if (idx + 1) % opt.print_freq == 0:
            #     print('Train: [{0}][{1}/{2}]\t'
            #           'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #           'loss {loss.val:.3f} ({loss.avg:.3f})\t'
            #           'Acc@1 {top1.val[0]:.3f} ({top1.avg[0]:.3f})'.format(
            #            epoch, idx + 1, len(train_loader), batch_time=batch_time,
            #            data_time=data_time, loss=losses, top1=top1))
            #     sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        with tqdm(range(val_loader.__len__())) as p_bar:
            for idx, (images, labels) in enumerate(val_loader):
                images = images.float().cuda()
                labels = labels.cuda()
                bsz = labels.shape[0]

                # forward
                feature = model.encoder(images)
                output = classifier(feature)
                loss = criterion(output, labels)

                # update metric
                losses.update(loss.item(), bsz)
                #acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                acc1 = accuracy(output, labels, topk=(1))
                top1.update(acc1[0], bsz)
                
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                #print(top1.val[0].item())
                p_bar.update(1)
                p_bar.set_description('test: [{}]'.format(idx))
                p_bar.set_postfix({"acc": top1.avg[0].item()})
                
                ## print info
                # if idx % opt.print_freq == 0:
                #     print('Test: [{0}/{1}]\t'
                #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                #           'Acc@1 {top1.val[0]:.3f} ({top1.avg[0]:.3f})'.format(
                #            idx, len(val_loader), batch_time=batch_time,
                #            loss=losses,top1=top1))

    #print(' * Acc@1 {top1.avg[0]:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()
    
    #checkpoint = torch.load('save/SupCon/path_linear_models/path_resnet50_lr_0.01_decay_0_bsz_128_220/ckpt_epoch_35.pth')
    #e = checkpoint['epoch']

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    # tensorboard
    #logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()

        print('total time {:.2f}, accuracy: {:.2f}'.format(time2 - time1, acc[0]))        
        # print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
        #     epoch, time2 - time1, acc[0]))

        # tensorboard logger
        #logger.log_value('train_loss', loss, epoch)
        #logger.log_value('train_acc', acc, epoch)
        #logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # eval for one epoch
        loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        #logger.log_value('val_loss', loss, epoch)
        #logger.log_value('val_acc', val_acc, epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            print('test accuracy: {:.2f}, best accuracy: {:.2f}'.format(val_acc[0], best_acc))
            save_model(classifier, optimizer, opt, epoch, save_file)
            #print('save model to {0}'.format(save_file))
        
        if val_acc[0] > best_acc:
            best_acc = val_acc[0]
            save_model(classifier, optimizer, opt, epoch, os.path.join(opt.save_folder, 'best.pth'))

    #print('best accuracy: {:.2f}'.format(best_acc))
    print('encoder path: {}'.format(opt.ckpt))
    print('best classifier saved in: {}'.format(os.path.join(opt.save_folder, 'best.pth')))


if __name__ == '__main__':
   main()
   #test()
   #exp()
