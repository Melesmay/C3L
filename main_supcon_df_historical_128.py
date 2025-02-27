import os

#.008 2 + (1 1) 8
#os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2'
# .05 4 (2 + 2) 8
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import sys
import argparse
import time
import math
import random


#import tensorboard_logger as tb_logger
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
#GaussianBlur,
from util import TwoCropTransform,  AverageMeter
from util import seed_torch
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet
from networks.xception import SupConXception
from losses import SCML
from dataloader import DFDataset

from tqdm import tqdm

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=50,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=5,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=120,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='80, 90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='xception')
    parser.add_argument('--dataset', type=str, default='celeb',
                        choices=['cifar10', 'cifar100', 'celeb', 'faceforensics'], help='dataset')
    parser.add_argument('--test_method', type=str, default = 'DF', choices=['DF', 'F2F', 'FSwap', 'FShifter', 'NT'])
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=128, help='parameter for RandomResizedCrop')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument('--harderPos', type=bool, default=True, help='weather apply harder positive')

    # temperature
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--use_device', type = str, default = '0')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
            help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='in128_raw_harder_query_epoch_his_v3_16_lam_8_2_4_mid_beta_08_08',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    #if opt.dataset == 'path':
        #print(opt.mean)
        #assert opt.data_folder is not None and opt.mean is not None and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'

    if opt.dataset == 'faceforensics':
        opt.model_path = './save/SupCon/{}_models/test/{}'.format(opt.dataset, opt.test_method)
        opt.tb_path = './save/SupCon/{}_tensorboard/test/{}'.format(opt.dataset, opt.test_method)

    else:
        opt.model_path = './save/SupCon/{}_models/abla'.format(opt.dataset)
        opt.tb_path = './save/SupCon/{}_tensorboard/abla'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.001
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

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'celeb':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif opt.dataset == 'faceforensics':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size = opt.size, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        #transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif opt.dataset == 'celeb':
        train_dataset = DFDataset(txt = 'datasets/Celeb-DF-v2/train_txt.txt', transform = train_transform)
    elif opt.dataset == 'faceforensics':
        load_txt = 'datasets/FaceForensics/test{}.txt'.format(opt.test_method)
        train_dataset = DFDataset(txt = load_txt, transform = train_transform)
        #train_dataset = DFDataset(txt = 'datasets/FaceForensics/train.txt', transform = train_transform)
    else:
        raise ValueError(opt.dataset)

    #print(train_dataset.__getitem__(10))

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last = True)

    return train_loader


def set_model(opt):
    
    if opt.model == 'xception':
        model = SupConXception()
        
    else:
        model = SupConResNet(name=opt.model)
    
    criterion = SCML(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    #save the sum of real in past epoch
    last_real = 0
    center = 0

    end = time.time()

    last_top_fake = []
    last_top_real = []

    with tqdm(range(train_loader.__len__())) as p_bar:
        for idx, (images, labels) in enumerate(train_loader):
            data_time.update(time.time() - end)
            
            labels = labels.clone().detach()

            images = torch.cat([images[0], images[1]], dim=0)
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]
            
            # warm-up learning rate
            warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

            # compute loss
            features = model(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            ### harder fake
            if opt.harderPos:
                #print(features.shape)
                ## 1 calculate current real center
                num_real = torch.sum(labels, dim = 0)
                #center_local = (torch.sum(torch.mul(f1.T, labels).T, dim = 0) / num_real + torch.sum(torch.mul(f2.T, labels).T, dim = 0) / num_real) / 2
                
                if idx == 0:
                    center = (torch.sum(torch.mul(f1.T, labels).T, dim = 0) / num_real + torch.sum(torch.mul(f2.T, labels).T, dim = 0) / num_real) / 2
                else:
                    #center = ((torch.sum(torch.mul(f1.T, labels).T, dim = 0) / num_real + torch.sum(torch.mul(f2.T, labels).T, dim = 0) / num_real) / 2) * a + center * (1 - a)
                    center = (((torch.sum(torch.mul(f1.T, labels).T, dim = 0) / num_real + torch.sum(torch.mul(f2.T, labels).T, dim = 0) / num_real) / 2) * num_real + center * last_real) / (num_real + last_real)
                #center_local = F.normalize(center_local, dim = 0)
                center = F.normalize(center, dim = 0)
                f = torch.cat((f1, f2), dim = 0)
                f_labels = torch.cat((labels, labels), 0)

                ## 2 calculate and sort similarity with center resperctively
                #cen_local_sim_f = torch.matmul(f, center_local.unsqueeze(1))
                cen_sim_f = torch.matmul(f, center.unsqueeze(1))

                real_sim_f, real_idx_f = torch.sort(torch.mul(cen_sim_f.T, f_labels), descending = True)
                #real_sim_f, real_idx_f = torch.sort(torch.mul(cen_local_sim_f.T, f_labels), descending = True)
                fake_sim_f, fake_idx_f = torch.sort(torch.mul(cen_sim_f.T, torch.abs(torch.sub(f_labels, 1))), descending = True)

                ## 3 select top-n features       
                real_nonz_midx_f = torch.max(torch.nonzero(real_sim_f)).item()
                fake_nonz_midx_f = torch.max(torch.nonzero(fake_sim_f)).item()

                k = 128
                m = 4 * epoch // 40 
                # top-n real
                #top_real_f = real_idx_f[:, : min(k, real_nonz_midx_f)][0]

                # top-n fake
                top_fake_f = fake_idx_f[:, : min(k, fake_nonz_midx_f)][0]

                # tail-n real
                tail_real_f = real_idx_f[:, max(real_nonz_midx_f - k, 0) : real_nonz_midx_f][0]

                # tail-n fake
                #tail_fake_f = fake_idx_f[:, max(fake_nonz_midx_f - k, 0) : fake_nonz_midx_f][0]

                ## nearest neighbour based
                # tail real
                real_nonz = real_sim_f[0][: real_nonz_midx_f]
                real_nonz_idx = real_idx_f[0][: real_nonz_midx_f]
                nn_real_len = 8
                nn_real = real_nonz_idx[len(real_nonz) - nn_real_len :]

                # top fake
                fake_nonz = fake_sim_f[0][: fake_nonz_midx_f]
                fake_nonz_idx = fake_idx_f[0][: fake_nonz_midx_f]
                #nn_fake_len = torch.sum(fake_nonz.gt(fake_nonz.mean()).int()).item()

                use_fr = 0

                nn_fake_len = 8
                nn_fake = fake_nonz_idx[: nn_fake_len]
                lam = 4

                #initualize
                if len(last_top_fake) == 0:
                    last_top_fake = f.index_select(0, nn_fake)
                    last_top_real = f.index_select(0, nn_real)


                ## 4 mix feature
                #a = random.random() + 1
                #a_rf = random.random() * 0.5

                a = random.betavariate(2, 1) + 1
                a_rf = random.betavariate(0.8, 0.8)

                ##v3 k-nearest based
                if len(nn_real):
                    #find m most likely real for the k least likely real
                    sim_rr = torch.matmul(last_top_real.detach(), torch.mul(f.T, f_labels))
                    
                    rr_sorted_sim, rr_index = torch.sort(sim_rr, descending = True)
                    rr_nonz_midx = torch.max(torch.nonzero(rr_sorted_sim)).item()
                    rr_nonz = rr_sorted_sim[:, 1: rr_nonz_midx]
                    rr_nonz_idx = rr_index[:, 1: rr_nonz_midx]
                    mean_rr = rr_nonz.mean(dim = 1)
                    k_rr = 2
                    
                    fm1 = last_top_real.detach().unsqueeze(1)
                    #fm1 = f.index_select(0, nn_real).unsqueeze(1)
                    #print(f.index_select(0, nn_real).unsqueeze(1))
                    fm2 = torch.stack(f.index_select(0, torch.cat(torch.unbind(rr_nonz_idx[:, :k_rr], dim = 0), dim = 0)).split(k_rr, dim = 0))

                    fm_rr = F.normalize(torch.mean(torch.cat((fm1, fm2), dim = 1), dim = 1), dim = 1).unsqueeze(1)
                    #print(fm_rr.shape)
                    features = torch.cat((features, torch.cat((fm_rr, fm_rr), dim = 1)), dim = 0)
                    labels = torch.cat((labels, torch.ones(nn_real_len).cuda()),dim = 0)
                    #print(features.shape)
                    #print(labels.shape)
                   

                if len(nn_fake):
                    #find k most likely real for the k most likely fake
                    sim_fr = torch.matmul(f.index_select(0, nn_fake), torch.mul(f.T, f_labels))

                    #find k most likely fake for the k most likely fake
                    sim_ff = torch.matmul(last_top_fake.detach(), torch.mul(f.T, torch.abs(torch.sub(f_labels, 1))))

                    ff_sorted_sim, ff_index = torch.sort(sim_ff, descending = True)
                    ff_nonz_midx = torch.max(torch.nonzero(ff_sorted_sim)).item()
                    ff_nonz = ff_sorted_sim[:, 1: ff_nonz_midx]
                    ff_nonz_idx = ff_index[:, 1: ff_nonz_midx]
                    mean_ff = ff_nonz.mean(dim = 1)
                    k_ff = 2

                    fm1 = last_top_fake.detach().unsqueeze(1)
                    #fm1 = f.index_select(0, nn_fake).unsqueeze(1)
                    fm2 = torch.stack(f.index_select(0, torch.cat(torch.unbind(ff_nonz_idx[:, :k_ff], dim = 0), dim = 0)).split(k_ff, dim = 0))

                    fm_ff = F.normalize(torch.mean(torch.cat((fm1, fm2), dim = 1), dim = 1), dim = 1).unsqueeze(1)
                    #print(fm_ff.shape)
                    features = torch.cat((features, torch.cat((fm_ff, fm_ff), dim = 1)), dim = 0)
                    labels = torch.cat((labels, torch.zeros(nn_fake_len).cuda()),dim = 0)
                    #print(features.shape)
                    #print(labels.shape)

                    
                    if not use_fr:
                        fr_sorted_sim, fr_index = torch.sort(sim_fr, descending = True)
                        fr_nonz_midx = torch.max(torch.nonzero(fr_sorted_sim)).item()
                        fr_nonz = fr_sorted_sim[:, 1: fr_nonz_midx]
                        fr_nonz_idx = fr_index[:, 1: fr_nonz_midx]
                        k_fr = 2

                        fm1 = f.index_select(0, nn_fake[: int(nn_fake_len / lam)]).unsqueeze(1)
                        fm2 = torch.stack(f.index_select(0, torch.cat(torch.unbind(fr_nonz_idx[: int(nn_fake_len / lam), :k_fr], dim = 0), dim = 0)).split(k_fr, dim = 0))

                        fm_fr = F.normalize(torch.cat(torch.unbind(torch.add(a_rf * fm1, (1 - a_rf) * fm2), dim = 1), dim = 0).unsqueeze(1), dim = 2)
                        #print(fm_fr.shape)
                        #print(fm1.shape)
                        #print(fm2.shape)
                        if k_fr * nn_fake_len:
                            features = torch.cat((features, torch.cat((fm_fr, fm_fr), dim = 1)), dim = 0)
                            labels = torch.cat((labels, torch.zeros(int(k_fr * nn_fake_len / lam)).cuda()),dim = 0)
                            #print(features.shape)
                            #print(labels.shape)
                    else:
                        k_fr = 0
                #print(features.shape)
                add_len = int(nn_real_len + k_fr * nn_fake_len / lam + nn_fake_len)
                
                last_real += num_real

                # update based on similarity
                if idx:
                    top_fake_all = torch.cat((f.index_select(0, nn_fake), last_top_fake), dim = 0)
                    top_real_all = torch.cat((f.index_select(0, nn_real), last_top_real), dim = 0)

                    sim_fake_all = torch.matmul(top_fake_all, center.unsqueeze(1)).T.squeeze(0)
                    sim_fake, idx_fake = torch.sort(sim_fake_all, descending = True)

                    #sim_real_all = torch.matmul(top_real_all, center_local.unsqueeze(1)).T.squeeze(0)
                    sim_real_all = torch.matmul(top_real_all, center.unsqueeze(1)).T.squeeze(0)
                    sim_real, idx_real = torch.sort(sim_real_all, descending = False)

                    last_top_fake = top_fake_all.index_select(0, idx_fake)[: nn_fake_len]
                    last_top_real = top_real_all.index_select(0, idx_real)[: nn_real_len]


            if opt.method == 'SupCon':
                if opt.harderPos == True:
                    loss = criterion(features, labels, harder = add_len, mode = 'query', mid = int(k_fr * nn_fake_len / lam))
                else:
                    loss = criterion(features, labels)
            elif opt.method == 'SimCLR':
                loss = criterion(features)
            else:
                raise ValueError('contrastive method not supported: {}'.
                                 format(opt.method))

            # update metric
            losses.update(loss.item(), bsz)

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
            p_bar.set_postfix({"loss": losses.val, "avg loss": losses.avg})

            # if (idx + 1) % opt.print_freq == 0:
            #     print('Train: [{0}][{1}/{2}]\t'
            #           'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #           'loss {loss.val:.5f} ({loss.avg:.5f})'.format(
            #            epoch, idx + 1, len(train_loader), batch_time=batch_time,
            #            data_time=data_time, loss=losses))
            #     #print('fake len {}, real len {}'.format(nn_fake_len, nn_real_len))
            #     sys.stdout.flush()



    return losses.avg


def main():
    opt = parse_option()

    #os.environ["CUDA_VISIBLE_DEVICES"] = opt.use_device

    '''
    # training from existing weight
    checkpoint = torch.load('save/SupCon/celeb_models/abla/SupCon_celeb_xception_lr_0.01_decay_0.0001_bsz_128_temp_0.1_trial__in128_raw_harder_query_epoch_his_v3_64_lam_8_2_4_mid_beta_08_08/ckpt_epoch_60.pth')
    opt = checkpoint['opt']
    
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    e = checkpoint['epoch']
    '''
    # build model and criterion
    model, criterion = set_model(opt)

    # build data loader
    train_loader = set_loader(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # tensorboard
    #logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        #logger.log_value('loss', loss, epoch)
        #logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    #seed_torch()    
    main()
