import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

class SeparableConv2d(nn.Module):
  def __init__(self, c_in, c_out, ks, stride=1, padding=0, dilation=1, bias=False):
    super(SeparableConv2d, self).__init__()
    self.c = nn.Conv2d(c_in, c_in, ks, stride, padding, dilation, groups=c_in, bias=bias)
    self.pointwise = nn.Conv2d(c_in, c_out, 1, 1, 0, 1, 1, bias=bias)

  def forward(self, x):
    x = self.c(x)
    x = self.pointwise(x)
    return x

class Block(nn.Module):
  def __init__(self, c_in, c_out, reps, stride=1, start_with_relu=True, grow_first=True):
    super(Block, self).__init__()
    
    self.skip = None
    self.skip_bn = None
    if c_out != c_in or stride!= 1:
      self.skip = nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False)
      self.skip_bn = nn.BatchNorm2d(c_out)

    self.relu = nn.ReLU(inplace=True)
    
    rep = []
    c = c_in
    if grow_first:
      rep.append(self.relu)
      rep.append(SeparableConv2d(c_in, c_out, 3, stride=1, padding=1, bias=False))
      rep.append(nn.BatchNorm2d(c_out))
      c = c_out
    
    for i in range(reps - 1):
      rep.append(self.relu)
      rep.append(SeparableConv2d(c, c, 3, stride=1, padding=1, bias=False))
      rep.append(nn.BatchNorm2d(c))

    if not grow_first:
      rep.append(self.relu)
      rep.append(SeparableConv2d(c_in, c_out, 3, stride=1, padding=1, bias=False))
      rep.append(nn.BatchNorm2d(c_out))
    
    if not start_with_relu:
      rep = rep[1:]
    else:
      rep[0] = nn.ReLU(inplace=False)

    if stride != 1:
      rep.append(nn.MaxPool2d(3, stride, 1))
    self.rep = nn.Sequential(*rep)

  def forward(self, inp):
    x = self.rep(inp)
    
    if self.skip is not None:
      y = self.skip(inp)
      y = self.skip_bn(y)
    else:
      y = inp
    
    x += y
    return x

class Xception(nn.Module):
  """
  Xception optimized for the ImageNet dataset, as specified in
  https://arxiv.org/pdf/1610.02357.pdf
  """
  def __init__(self, num_classes=1000):
    super(Xception, self).__init__()
    self.num_classes = num_classes

    self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
    self.bn1 = nn.BatchNorm2d(32)
    self.relu = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(32,64,3,bias=False)
    self.bn2 = nn.BatchNorm2d(64)

    self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
    self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
    self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

    self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

    self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
    self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

    self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

    self.conv3 = SeparableConv2d(1024,1536,3,1,1)
    self.bn3 = nn.BatchNorm2d(1536)

    self.conv4 = SeparableConv2d(1536,2048,3,1,1)
    self.bn4 = nn.BatchNorm2d(2048)
    
    # self.last_linear = nn.Linear(2048, num_classes)

    
  def features(self, input):
    x = self.conv1(input)
    x = self.bn1(x)
    x = self.relu(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)

    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)
    x = self.block7(x)
    x = self.block8(x)
    x = self.block9(x)
    x = self.block10(x)
    x = self.block11(x)
    x = self.block12(x)

    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu(x)

    x = self.conv4(x)
    x = self.bn4(x)
    return x

  def logits(self, features):
    x = self.relu(features)
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = x.view(x.size(0), -1)
    # x = self.last_linear(x)
    return x

  def forward(self, input):
    x = self.features(input)
    x = self.logits(x)
    return x

def init_weights(m):
  classname = m.__class__.__name__
  if classname.find('SeparableConv2d') != -1:
    m.c.weight.data.normal_(0.0, 0.01)
    if m.c.bias is not None:
      m.c.bias.data.fill_(0)
    m.pointwise.weight.data.normal_(0.0, 0.01)
    if m.pointwise.bias is not None:
      m.pointwise.bias.data.fill_(0)
  elif classname.find('Conv') != -1 or classname.find('Linear') != -1:
    m.weight.data.normal_(0.0, 0.01)
    if m.bias is not None:
      m.bias.data.fill_(0)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.01)
    m.bias.data.fill_(0)
  elif classname.find('LSTM') != -1:
    for i in m._parameters:
      if i.__class__.__name__.find('weight') != -1:
        i.data.normal_(0.0, 0.01)
      elif i.__class__.__name__.find('bias') != -1:
        i.bias.data.fill_(0)

# model_dicts = {
#   'xception': [Xception, 2048]
# }

class SupConXception(nn.Module):
  """backbone + projection head"""
  def __init__(self, head='mlp', feat_dim=128):
    super(SupConXception, self).__init__()
    #if load_pretrain:
    #model = Xception().state_dict()
    
    model = Xception()
    dim_in = 2048
    
    # model_dict = model_fun.state_dict()

    # state_dict = torch.load('networks/xception-b5690688.pth')

    # for name, weights in state_dict.items():
    #    if 'pointwise' in name:
    #      state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)

    # model_dict.update(state_dict)
    # model_fun.load_state_dict(model_dict)
    state_dict = torch.load('networks/xception-b5690688.pth')
    for name, weights in state_dict.items():
      if 'pointwise' in name:
        state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    #del state_dict['fc.weight']
    #del state_dict['fc.bias']
    model.load_state_dict(state_dict, False)

    self.encoder = model

    if head == 'linear':
      self.head = nn.Linear(dim_in, feat_dim)   
    elif head == 'mlp':
      self.head = nn.Sequential(
        nn.Linear(dim_in, dim_in),
        nn.ReLU(inplace=True),
        nn.Linear(dim_in, feat_dim)
      )
    else:
      raise NotImplementedError(
        'head not supported: {}'.format(head))

  def forward(self, x):
    feat = self.encoder(x)
    feat = F.normalize(self.head(feat), dim=1)
    return feat

class SupCEXception(nn.Module):
    """encoder + classifier"""
    def __init__(self, num_classes=2):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, num_classes=2):
        super(LinearClassifier, self).__init__()
        #_, feat_dim = model_dicts[name]
        feat_dim = 2048
        #self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(feat_dim, num_classes)

    def logits(self, features):
        x = self.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, features):
        return self.fc(features)
        # return self.logits(features)

class Model:
  def __init__(self, maptype='None', templates=None, num_classes=2, load_pretrain=True):
    model = Xception(maptype, templates, num_classes=num_classes)
    if load_pretrain:
      state_dict = torch.load('./xception-b5690688.pth')
      for name, weights in state_dict:
        if 'pointwise' in name:
          state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
      del state_dict['fc.weight']
      del state_dict['fc.bias']
      model.load_state_dict(state_dict, False)
    else:
      model.apply(init_weights)
    self.model = model

  def save(self, epoch, optim, model_dir):
    state = {'net': self.model.state_dict(), 'optim': optim.state_dict()}
    torch.save(state, '{0}/{1:06d}.tar'.format(model_dir, epoch))
    print('Saved model `{0}`'.format(epoch))

  def load(self, epoch, model_dir):
    filename = '{0}{1:06d}.tar'.format(model_dir, epoch)
    print('Loading model from {0}'.format(filename))
    if os.path.exists(filename):
      state = torch.load(filename)
      self.model.load_state_dict(state['net'])
    else:
      print('Failed to load model from {0}'.format(filename))

