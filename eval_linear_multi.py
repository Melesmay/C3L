from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
#import tensorboard_logger as tb_logger
from torchvision import transforms, datasets

import numpy

from dataloader import MyDataSet, ExpDataSet
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
#from networks.resnet_big import SupConResNet, LinearClassifier
from networks.xception import SupConXception, LinearClassifier

from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

#torch.backends.cudnn.enabled = False

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=250,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='celeb',
                        choices=['cifar10', 'cifar100', 'path', 'celeb', 'laohuang'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt_enc', type=str, default= \
            #'save/SupCon/path_linear_models/path_resnet50_lr_0.1_decay_0_bsz_128_180/ckpt_epoch_180.pth', \
            #'save/SupCon/celeb_models/abla/SupCon_celeb_xception_lr_0.01_decay_0.0001_bsz_128_temp_0.1_trial_t-s_1-2_harder_False/ckpt_epoch_100.pth',\
            'save/SupCon/faceforensics_models/test/FShifter/SupCon_faceforensics_xception_lr_0.05_decay_0.0001_bsz_128_temp_0.1_trial_SupCon_FSt/ckpt_epoch_85.pth',\
            #'save/SupCon/faceforensics_models/FSwap/SupCon_faceforensics_xception_lr_0.1_decay_0.0001_bsz_128_temp_0.1_trial_in128_raw_harder_query_epoch_his_v3_16_lam_4_2_4_mid_beta_08_08/ckpt_epoch_100.pth',\
                        help='path to pre-trained encoder')
    parser.add_argument('--ckpt_clf', type=str, default= \
            'save/SupCon/faceforensics_linear_models/faceforensics_xception_lr_0.01_decay_0_bsz_256_trial_FSt_85/best.pth', \
            #'save/SupCon/faceforensics_linear_models/faceforensics_xception_lr_0.01_decay_0_bsz_256_trial_in128_testFSp_False_100/best.pth',\
                        help='path to pre-trained classifier')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_linear_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_linear_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}_250'.\
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
    elif opt.dataset == 'path' or opt.dataset == 'celeb' or opt.dataset == 'laohuang':
        opt.n_cls = 2
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt

def set_loader(opt, train_txt = 'datasets/Celeb-DF-v2/train_txt.txt', test_txt = 'datasets/Celeb-DF-v2/test_txt.txt', PATH = 'C:/Users/admin/Desktop/Celeb-DF-v2-face/'):
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
    elif opt.dataset == 'celeb':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    elif opt.dataset == 'laohuang':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        #transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128,128)),
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

    classifier = LinearClassifier(num_classes=opt.n_cls)

    #ckpt = torch.load(opt.ckpt, map_location='cpu')
    #state_dict = ckpt['model']
    
    ckpt = torch.load(opt.ckpt_enc)
    state_dict = ckpt['model']
    class_ckpt = torch.load(opt.ckpt_clf)
    class_state_dict = class_ckpt['model']
    #classifier.load_state_dict(class_state_dict)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
        
        #print(state_dict)

        model.load_state_dict(state_dict)
        classifier.load_state_dict(class_state_dict)

    return model, classifier, criterion

def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    auc = AverageMeter()
    num = 0
    label = []
    logit_0 = []
    logit_1 = []

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]
            num += bsz

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

            logit_0 = numpy.concatenate((logit_0, output.cpu().numpy()[:, 0]), axis = 0)
            logit_1 = numpy.concatenate((logit_1, output.cpu().numpy()[:, 1]), axis = 0)
            label = numpy.concatenate((label, labels.cpu().numpy()), axis = 0)

            #print(top1.val[0].item())
            
            '''
            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val[0]:.3f} ({top1.avg[0]:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses,top1=top1))
            '''
    print(' * Acc {top1.avg[0]:.4f}'.format(top1=top1))
    return losses.avg, top1.avg, num, logit_0, logit_1, label

def test(train_txt = 'datasets/FaceForensics/temp.txt', test_txt = 'datasets/Celeb-DF-v2/test_txt.txt', PATH = 'C:/Users/admin/Desktop/Celeb-DF-v2-face/'):
    opt = parse_option()
    #opt.batch_size = int(opt.batch_size / 2)
    
    _, test_loader = set_loader(opt, train_txt, test_txt, PATH)

    model, classifier, criterion = set_model(opt)
    loss, test_acc, n, logit_0, logit_1, label = validate(test_loader, model, classifier, criterion, opt)
    return loss, test_acc, n, logit_0, logit_1, label



def set_exploader(opt, path):
    if opt.dataset == 'celeb':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        
        normalize = transforms.Normalize(mean=mean, std=std)

        val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299,299)),
            transforms.ToTensor(),
            normalize,
            ])

        #val_dataset = ExpDataSet('../val_label.txt', transform = val_transform)
        test_dataset = ExpDataSet(path,transform = val_transform)
        
        # val_loader = torch.utils.data.DataLoader(
        #     val_dataset, batch_size=64, shuffle=False,
        #     num_workers=16, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=64, shuffle=False,
            num_workers=16, pin_memory=True)
        #return val_loader, test_loader
        return test_loader

def exp_val(val_loader, model, classifier, criterion, opt, attri):
    """validation"""    
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, images in enumerate(val_loader):
            images = images.float().cuda()

            # forward
            output = classifier(model.encoder(images))
            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()
            
            for i in range(pred.numel()):
                out = pred.cpu().detach().numpy().tolist()[0][i]
                write_txt(out, attri)

def write_txt(pred, attri):
    with open("{0}_output_100.txt".format(attri),"a")as f:
        f.write(('{0}{1}').format(pred,'\n'))

def exp():
    opt = parse_option()

    val_loader, test_loader = set_exploader(opt, 'dataset/laohuang/pyq_0.txt')

    model, classifier, criterion = set_model(opt)

    exp_val(val_loader, model, classifier, criterion, opt, 'val')
    exp_val(test_loader, model, classifier, criterion, opt, 'test')
    print('done.')

def t_sne():
    opt = parse_option()
    _, test_loader = set_loader(opt)
    opt.ckpt = 'save/SupCon/celeb_linear_models/celeb_xception_lr_0.01_decay_0_bsz_128_0_alt/ckpt_epoch_20.pth'
    #print(opt.ckpt)
    model, classifier, criterion = set_model(opt)

    model.eval()
    classifier.eval()

    fea = []
    lab = []

    tsne = TSNE(n_components = 2, init = 'pca', perplexity = 30)
    plt.cla()
    
    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            feature = model.encoder(images)
            #output = classifier(feature)
            #loss = criterion(output, labels)

            low_demision = tsne.fit_transform(features)

            
            X, Y = low_demision[:, 0], low_demision[:, 1]

            for x, y, s in zip(X, Y, labels):
                c = cm.rainbow(int(255/2 * s))
                plt.text(x, y, s, backgroundcolor = c, fontsize = 9)
            plt.xlim(X.min(), X.max())
            plt.ylim(Y.min(), Y.max())
        plt.savefig('tsne.jpg')

def normalization(data):
    _range = numpy.max(data) - numpy.min(data)
    return (data - numpy.min(data)) / _range


def test_all(data_com, test_txt):
    for j in range(len(data_com)):
        print(data_com[j])
        acc_total = 0
        num_total = 0

        labels = []
        logits_0 = []
        logits_1 = []

        for i in range(len(test_txt)):
            print(test_txt[i].split('/')[2].split('.')[0])
            loss, test_acc, n, logit_0, logit_1, label = test(test_txt=test_txt[i], PATH=data_com[j])
        # calculate total acc
            acc_total += test_acc * n
            num_total += n

            labels = numpy.concatenate((labels, label), axis = 0)
            logits_0 = numpy.concatenate((logits_0, logit_0), axis = 0)
            logits_1 = numpy.concatenate((logits_1, logit_1), axis = 0)

        print('total acc {acc:.4f}'.format(acc = (acc_total / num_total).item()))

        auc_0 = roc_auc_score(labels, normalization(logits_0))
        auc_1 = roc_auc_score(labels, normalization(logits_1))
        print('auc 0: {}, auc 1: {}'.format(auc_0, auc_1, '.6f'))

def test_single(data_com, test_txt):
    real_logits_0 = []
    real_logits_1 = []
    real_label = []
    for j in range(len(data_com)):
        print(data_com[j])
        #acc_total = 0
        #num_total = 0

        labels = []
        logits_0 = []
        logits_1 = []

        for i in range(len(test_txt)):
            print(test_txt[i].split('/')[2].split('.')[0])
            loss, test_acc, n, logit_0, logit_1, label = test(test_txt=test_txt[i], PATH=data_com[j])
        # calculate total acc
            #acc_total += test_acc * n
            #num_total += n

            if test_txt[i].split('/')[2].split('.')[0] == 'youtube':
                real_label = label
                real_logits_0 = logit_0
                real_logits_1 = logit_1
            else:
                labels = numpy.concatenate((real_label, label), axis = 0)
                logits_0 = numpy.concatenate((real_logits_0, logit_0), axis = 0)
                logits_1 = numpy.concatenate((real_logits_1, logit_1), axis = 0)

                auc_0 = roc_auc_score(labels, normalization(logits_0))
                auc_1 = roc_auc_score(labels, normalization(logits_1))
                print('auc 0: {}, auc 1: {}'.format(auc_0, auc_1, '.6f'))

        
if __name__ == '__main__':
   #for i in range(5):
   '''
   data_com = ['C:/Users/admin/Desktop/Celeb-DF-v2-face/', 'Z:/DataSets/Celeb-DF-v2-compress/c23/images/', 'Z:/DataSets/Celeb-DF-v2-compress/c40/images/']
   test_txt = ['datasets/Celeb-DF-v2/celeb_synthesis.txt', 'datasets/Celeb-DF-v2/celeb_real.txt', 'datasets/Celeb-DF-v2/youtube_real.txt']
   
   '''
   data_com = ['../../../data/FaceForensics++_face/']
   test_txt = ['datasets/FaceForensics/youtube.txt', 'datasets/FaceForensics/Deepfakes.txt', 'datasets/FaceForensics/Face2Face.txt', 'datasets/FaceForensics/FaceShifter.txt', 'datasets/FaceForensics/FaceSwap.txt', 'datasets/FaceForensics/NeuralTextures.txt']
   
   test_single(data_com, test_txt)
   
   #exp()
