
import os

from imageio import imread

import ast
import re
import time

import torch.nn
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

PATH = '../../../data/FaceForensics++_face/'
#PATH = '../../../data/Celeb-DF-v2-face/'
#PATH = '../../../data/Celeb-DF-v2-compress/c29/images/'


def def_loader(image_pair):
    #print(PATH + image_pair[0])
    return [imread(PATH + image_pair[0]), imread(PATH + image_pair[1])]

def loader(image_pair):
    image0 = imread(PATH + image_pair[0])
    image1 = imread(PATH + image_pair[0])

    return[image0, image1]

def sin_loader(image):
    return imread(PATH + image)

def data_pairing(images):

    new_images = []

    real_images = []
    fake_images = []

    files = []
    attr = []

    #print('constructing positive pair...')
    
    for i in range(0, len(images)):
        #print(images[i][0])
        file = images[i][0] 
        files.append(file.split('/'))
        #print(files[i])
        attr.append(files[i][1].split('_'))
        #print(attr)
    
    #print(len(images))
# construct positive pair
# inner_frame
    
        if images[i][1] == 1:
            #print(images[i][0])
            real_images.append([[images[i][0], images[i][0]], images[i][1]])
        else:
            fake_images.append([[images[i][0], images[i][0]], images[i][1]])

    len_base = min(len(real_images), len(fake_images))
    #print(len_base)
    print('inner frame real pairs {0}, inner video fake pairs {1}, base length {2}'.format(len(real_images), len(fake_images), len_base))
    #new_images.extend(random.sample(real_images, int(len_base / 17 * 1)))
    #new_images.extend(random.sample(fake_images, int(len_base / 17 * 1)))    

    #new_images.extend(random.sample(real_images, int(len_base / 2 )))
    #new_images.extend(random.sample(fake_images, int(len_base / 2 ))) 

    new_images.extend(random.sample(real_images, len_base))
    new_images.extend(random.sample(fake_images, len_base))    

    
# inner-video
    # celeb-df
    '''
    real_images = []
    fake_images = []
    i = 0
    m = 8
    while i < len(images):
        video = files[i][1]
        #print(video)
        c_frame = 1
        while i + c_frame < len(files):
            if video == files[i + c_frame][1]:
                c_frame += 1
            else:
                break
        #print(video)
        #print(c)
        
        for  j in range(0, c_frame - m):
            #p1 = '{0}/{1}/{2}/{3}/{4}/{5}.png'.format(files[i][0],files[i][1],files[i][2],files[i][3], video, j)
            #p2 = '{0}/{1}/{2}/{3}/{4}/{5}.png'.format(files[i][0],files[i][1],files[i][2],files[i][3], video, j + m)

            p1 = '{0}/{1}/{2}.png'.format(files[i][0], video, j)
            #print(p1)
            p2 = '{0}/{1}/{2}.png'.format(files[i][0], video, j + m)

            if images[i][1] == 1:
                if os.path.exists('{0}{1}'.format(PATH, p1)) and os.path.exists(PATH + p2):
                    real_images.append([[p1, p2], images[i][1]])
            else:
                if os.path.exists('{0}{1}'.format(PATH, p1)) and os.path.exists(PATH + p2):
                    fake_images.append([[p1, p2], images[i][1]]) 
            j += m 
        i = i + c_frame
    #print(real_images[len(real_images)-1])
    print('inner video real pairs {0}, inner video fake pairs {1}'.format(len(real_images), len(fake_images)))
    len_tempory = min(int(len_base /9 * 8), min(len(real_images), len(fake_images)))
    new_images.extend(random.sample(real_images, int(len_tempory)))
    new_images.extend(random.sample(fake_images, int(len_tempory)))
    '''
    
    # ff++
    '''
    real_images = []
    fake_images = []
    i = 0
    m = 8
    while i < len(images):
        video = files[i][4]
        c_frame = 1
        #print(len(files))
        while i + c_frame < len(files):
            if video == files[i + c_frame][4]:
                c_frame += 1
            else:
                break
        #print(video)
        
        for  j in range(0, c_frame - m):
            p1 = '{0}/{1}/{2}/{3}/{4}/{5}.png'.format(files[i][0], files[i][1], files[i][2], files[i][3], video, j)
            p2 = '{0}/{1}/{2}/{3}/{4}/{5}.png'.format(files[i][0], files[i][1], files[i][2], files[i][3], video, j + m)
            #print(p1)
            if images[i][1] == 1:
                if os.path.exists('{0}{1}'.format(PATH, p1)) and os.path.exists(PATH + p2):
                    real_images.append([[p1, p2], images[i][1]])
            else:
                if os.path.exists('{0}{1}'.format(PATH, p1)) and os.path.exists(PATH + p2):
                    fake_images.append([[p1, p2], images[i][1]]) 
            j += m + 1
        i = i + c_frame
    #print(real_images[len(real_images)-1])
    print('inner video real pairs {0}, inner video fake pairs {1}'.format(len(real_images), len(fake_images)))
    len_tempory = min(int(len_base / 2), len(real_images))
    new_images.extend(random.sample(real_images, len_tempory))
    new_images.extend(random.sample(fake_images, len_tempory))
    '''
    '''
## cross video relationships v2
# inner source id
    real_images = []
    #fake_images = []
    inner_source_list = []
    inner_target_list = []
    chk_video = []
    for i in range(0, len(images)):
        chk_video.append(files[i][1])

    
    #Celeb-real
    #video_list = []
    r_c = 100
    for pid in range(0, 61):
        for vid in range(0, 15):
            random.seed(pid+vid)
            if random.random() * 5 < 1:
                continue

            v1 = 'id{0}_000{1}'.format(pid, vid)
            if v1 in chk_video:
                match_source = list(set([i for i in chk_video if re.compile('id{0}_000\d'.format(pid, vid)).match(i)]))
                for i in range(0, len(match_source)):
                    v2 = match_source[i]
                    if v2 != v1:
                        cframe1 = len(os.listdir('{0}Celeb-real/{1}/'.format(PATH, v1)))
                        cframe2 = len(os.listdir('{0}Celeb-real/{1}/'.format(PATH, v2)))
                        for i in range(0, r_c):
                            random.seed(i)
                            p1 = 'Celeb-real/{0}/{1}.png'.format(v1, int(random.random() * cframe1))
                            p2 = 'Celeb-real/{0}/{1}.png'.format(v2, int(random.random() * cframe2))
                            if os.path.exists(PATH + p1) and os.path.exists(PATH + p2):
                                #print(p1, p2)
                                real_images.append([[p1, p2], 1])
          
    #Celeb-synthesis
    #video_list = []
    for sid in range(0, 61):
        for tid in range(0, 61):
            for vid in range(0, 15):                
                v1 = 'id{0}_id{1}_000{2}'.format(sid, tid, vid)
                random.seed(sid+tid+vid)
                if random.random() * 4 < 1:
                    continue
                if v1 in chk_video:
                    #inner source id
                    match_source = list(set([i for i in chk_video if re.compile('id{0}_id\d+_000{1}'.format(sid, vid)).match(i)]))
                    for i in range(0, len(match_source), 5):
                        v2 = match_source[i]
                        if v2 != v1:
                            #v2 = match_source[i]
                            cframe2 = len(os.listdir('{0}Celeb-synthesis/{1}/'.format(PATH, v2)))
                            for j in range(0, r_c):
                                random.seed(j)
                                frame = int(random.random() * min(cframe1,cframe2))
                                p1 = 'Celeb-synthesis/{0}/{1}.png'.format(v1, frame)
                                p2 = 'Celeb-synthesis/{0}/{1}.png'.format(v2, frame)
                                if os.path.exists(PATH + p1) and os.path.exists(PATH + p2):
                                    #print(p1, p2)
                                    inner_source_list.append([[p1, p2], 0])

                    #inner target id
                    match_source = list(set([i for i in chk_video if re.compile('id\d+_id{0}_000{1}'.format(tid, vid)).match(i)]))
                    for i in range(0, len(match_source), 5):
                        v3  = match_source[i]
                        if v3 != v1:
                            cframe3 = len(os.listdir('{0}Celeb-synthesis/{1}/'.format(PATH, v3)))
                            for j in range(0, r_c):
                                random.seed(j)
                                frame = int(random.random() * min(cframe1,cframe3))
                                p1 = 'Celeb-synthesis/{0}/{1}.png'.format(v1, frame)
                                p2 = 'Celeb-synthesis/{0}/{1}.png'.format(v3, frame)
                                if os.path.exists(PATH + p1) and os.path.exists(PATH + p2):
                                    #print(p1,p2)
                                    inner_target_list.append([[p1, p2], 0])

    #len_ref = min(min(len(real_images), 2 * min(min(len(inner_source_list), len(inner_target_list)), int(len_base / 8))), int(len_base / 2))
    len_ref = int(len_base / 9 * 1)
    print('inner source real pairs {0}, inner source fake pairs {1}, inner target fake pairs {2} reference length {3}'.format(len(real_images), len(inner_source_list), len(inner_target_list), len_ref))
    #new_images.extend(random.sample(real_images, len_ref))
    new_images.extend(random.sample(inner_source_list, int(len_ref )))
    new_images.extend(random.sample(inner_target_list, int(len_ref )))
    '''
    '''
##cross video relationship v1
    real_images = []
    fake_images = [] 
# inner-id
    chk_video = []
    for i in range(0, len(images)):
        chk_video.append(files[i][1])
    #print(chk_video)
    

    #Celeb-real
    video_list = []
    for pid in range(0, 61):
        for vid in range(0, 9):
            video_name = 'id{0}_000{1}'.format(pid, vid)
            #print(video_name)
            if video_name in chk_video:
                #print('exist')
                cframe = len(os.listdir('{0}Celeb-real/{1}/'.format(PATH, video_name)))
                #print(video_name, cframe)
                video_list.append([video_name, cframe])

    #for i in range(len(video_list) - 1):
    i = 0
    while i < len(video_list) - 1:
        random.seed(i)
        i += int(random.random() * 50)
        if i+1 <len(video_list):
            if video_list[i][0].split('_')[0] == video_list[i + 1][0].split('_')[0]:
                for i_frame in range(video_list[i][1]):
                    random.seed(i+i_frame)
                    dis = int(random.random() * video_list[i + 1][1])

                    if os.path.exists('{0}Celeb-real/{1}/{2}.png'.format(PATH, video_list[i + 1][0], dis)) and os.path.exists('{0}Celeb-real/{1}/{2}.png'.format(PATH, video_list[i][0], i_frame)):
                        p1 = 'Celeb-real/{0}/{1}.png'.format(video_list[i][0], i_frame)
                        p2 = 'Celeb-real/{0}/{1}.png'.format(video_list[i+1][0], dis)
                        #print(p1, p2)
                        real_images.append([[p1, p2], 1])
        i += 1

    #Celeb-synthesis
    video_list = []
    for sid in range(0, 61):
        for tid in range(0, 61):
            for vid in range(0, 9):
                video_name = 'id{0}_id{1}_000{2}'.format(tid, sid, vid)
                if video_name in chk_video:
                    #print(video_name)
                    cframe = len(os.listdir('{0}Celeb-synthesis/{1}/'.format(PATH, video_name)))
                    video_list.append([video_name, cframe])
    
    #for i in range(len(video_list) - 1):
    i = 0
    while i < len(video_list) - 1:
        random.seed(i)
        i += int(random.random() * 50)

        if i > len(video_list) - 1:
            break

        if video_list[i][0].split('_')[1] == video_list[i + 1][0].split('_')[1]:
            for i_frame in range(video_list[i][1]):
                random.seed(i + i_frame)
                dis = int(random.random() * video_list[i + 1][1])
                if not os.path.exists('{0}Celeb-synthesis/{1}/{2}.png'.format(PATH, video_list[i + 1][0], dis)):
                    dis += 1
                if os.path.exists('{0}Celeb-synthesis/{1}/{2}.png'.format(PATH, video_list[i + 1][0], dis)) and os.path.exists('{0}Celeb-synthesis/{1}/{2}.png'.format(PATH, video_list[i][0], i_frame)):
                    p1 = 'Celeb-synthesis/{0}/{1}.png'.format(video_list[i][0], i_frame)
                    p2 = 'Celeb-synthesis/{0}/{1}.png'.format(video_list[i + 1][0], dis)

                    fake_images.append([[p1, p2], 0])
        i += 1

    len_ref = min(min(len(real_images), len(fake_images)), int(len_base / 4))
    print('inner id real pairs {0}, inner id fake pairs {1}, reference length {2}'.format(len(real_images), len(fake_images), len_ref))
    new_images.extend(random.sample(real_images, len_ref))
    new_images.extend(random.sample(fake_images, len_ref))

    real_images = []
    fake_images = []

# outer_id
    #celeb-synthesis
    i = 0
    while i < len(video_list) - 1:
        random.seed(i)
        i += int(random.random() * 50)
        
        if i > len(video_list) - 1:
            break

        gap = 0
        #print(video_list[i][0].split('_')[1])
        while (video_list[i][0].split('_')[1] == video_list[i + gap][0].split('_')[1] or video_list[i][0].split('_')[0] == video_list[i + gap][0].split('_')[0]) and gap < len(video_list):
                gap += 1
                if i + gap >= len(video_list):
                    gap = -1
                    break
        if gap > -1:
            #print(video_list[i], video_list[i+gap])
            for i_frame in range(video_list[i][1]):
                random.seed(i+i_frame)
                dis = int(random.random() * video_list[i + gap][1])
                if not os.path.exists('{0}Celeb-synthesis/{1}/{2}.png'.format(PATH, video_list[i + 1][0], dis)):
                    dis += 1
                
                p1 = 'Celeb-synthesis/{0}/{1}.png'.format(video_list[i][0], i_frame)
                p2 = 'Celeb-synthesis/{0}/{1}.png'.format(video_list[i + 1][0], dis)
                if os.path.exists('{0}Celeb-synthesis/{1}/{2}.png'.format(PATH, video_list[i + 1][0], dis)) and os.path.exists('{0}Celeb-synthesis/{1}/{2}.png'.format(PATH, video_list[i][0], i_frame)):
                    fake_images.append([[p1, p2], 0])
                #print(p1, p2)
        i += 1

    #Celeb-real
    video_list = []
    for vid in range(0, 9):
        for pid in range(0, 61):
            video_name = 'id{0}_000{1}'.format(pid, vid)
            #print(video_name)
            if video_name in chk_video:
                cframe = len(os.listdir('{0}Celeb-real/{1}/'.format(PATH, video_name)))
                #print(video_name)
                video_list.append([video_name, cframe])
    print('complite loading celeb real video list')

    #for i in range(len(video_list) - 1):
    i = 0
    while i < len(video_list) - 1:
        random.seed(i)
        i += int(random.random() * 50)
        
        if i > len(video_list) - 1:
            break

        if video_list[i][0].split('_')[0] != video_list[i + 1][0].split('_')[0]:
            for i_frame in range(video_list[i][1]):
                random.seed(i + i_frame)
                dis = int(random.random() * video_list[i + 1][1])

                p1 = 'Celeb-real/{0}/{1}.png'.format(video_list[i][0], i_frame)
                p2 = 'Celeb-real/{0}/{1}.png'.format(video_list[i+1][0], dis)
                if os.path.exists('{0}Celeb-real/{1}/{2}.png'.format(PATH, video_list[i + 1][0], dis)) and os.path.exists('{0}Celeb-real/{1}/{2}.png'.format(PATH, video_list[i][0], i_frame)):
                    #p1 = 'Celeb-real/{0}/{1}.png'.format(video_list[i][0], i_frame)
                    #p2 = 'Celeb-real/{0}/{1}.png'.format(video_list[i+1][0], dis)
                    #print(p1, p2)
                    real_images.append([[p1, p2], 1])
        i += 1

    #YouTube-real
    video_list = []
    for vid in range(0, 299):
        video_name = '%05d' % vid
        if video_name in chk_video:
            cframe = len(os.listdir('{0}YouTube-real/{1}/'.format(PATH, video_name)))
            #print(cframe)
            video_list.append([video_name, cframe])
    
    #for i in range(len(video_list) - 1):
    i = 0
    while i < len(video_list) - 1:
        random.seed(i)
        i += int(random.random() * 20)

        if i > len(video_list) - 1:
            break

        for i_frame in range(video_list[i][1]):
            random.seed(i + i_frame)
            dis = int(random.random() * video_list[i + 1][1])
            
            if os.path.exists('{0}YouTube-real/{1}/{2}.png'.format(PATH, video_list[i + 1][0], dis)) and os.path.exists('{0}YouTube-real/{1}/{2}.png'.format(PATH, video_list[i][0], i_frame)):
                p1 = 'YouTube-real/{0}/{1}.png'.format(video_list[i][0], i_frame)
                p2 = 'YouTube-real/{0}/{1}.png'.format(video_list[i+1][0], dis)
                #print(p1, p2)
                real_images.append([[p1, p2], 1])
        i += 1 
    
    len_ref = min(min(len(real_images), len(fake_images)), int(len_base / 3))
    print('outer id real pairs {0}, outer id fake pairs {1}, reference length {2}'.format(len(real_images), len(fake_images), len_ref))
    
    new_images.extend(random.sample(real_images, len_ref))
    new_images.extend(random.sample(fake_images, len_ref))
    '''

    #for i in range(0, len(new_images)):
    #    print(new_images[i])
    
    #with open('loadtxt.txt', 'w') as f:
    #    for i in range(0, len(new_images)):
    #        print(new_images[i])
    #        f.write(new_images[i])
    return new_images


class DFDataset(Dataset):
    def __init__(self, txt, transform = None, target_transform = None, loader = def_loader):
        super(DFDataset, self).__init__()

        imgs = []

        # load from txt
        '''
        with open('load_final_0.txt', 'r') as f:
            line = f.readline().strip('\n')
            while line: 
                if line.find('[[') > -1:
                    ## image pair
                    imgs.append(ast.literal_eval(line))

                    ## single image
                    #imgs.append((ast.literal_eval(line)[0][0], ast.literal_eval(line)[1]))

                line = f.readline().strip('\n')

        
        # load from file
        '''
        with open(txt, 'r') as f:
            line = f.readline().strip('\n')
            while line:
                if line.strip() != '':
                    label_idx, img_path = line.split(' ')
                    
                    img_file = '{0}{1}'.format(PATH, img_path)
                    pic_list = os.listdir(img_file)
                    for i, pic in enumerate(pic_list):
                        
                        if int(pic.split('.')[0])%1 == 0:
                            #print(int(pic.split('.')[0]))
                            imgs.append(('{0}/{1}'.format(img_path, pic), int(label_idx)))
                    line = f.readline().strip('\n')
      
        imgs = data_pairing(imgs)
        
        self.imags = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, item):

        #print(self.imags[item])
        img_pair, label_idx = self.imags[item]
        
        img_pair = self.loader(img_pair)
        if self.transform is not None:
        #    img_pair = self.transform(img_pair)
            img_pair = [self.transform(img_pair[0]), self.transform(img_pair[1])]
        return img_pair, label_idx

    def __len__(self):
        return len(self.imags)

def linear_loader(image_pair):
    
    image = imread(image_pair)
    
    return image
'''
class MyDataSet(Dataset):
    def __init__(self, txt, loader = linear_loader, transform = None):
        super(MyDataSet, self).__init__()

        # txt = '{0}{1}'.format(path, txt)
        imgs = []
        with open(txt, 'r') as f:
            line = f.readline().strip('\n')
            while line:
                if line.strip() != '':
                    if ' ' in line:
                        label_idx, img_path = line.split(' ')
                        #print('{0}{1}'.format(PATH, img_path))
                        
                        #celeb
                        img_file = '{0}{1}'.format(PATH, img_path)
                        pic_list = os.listdir(img_file)
                        for i, pic in enumerate(pic_list):
                            #print('{0}/{1}'.format(img_file, pic))
                            imgs.append(('{0}/{1}'.format(img_file, pic), int(label_idx)))
                        
                        #imgs.append(('{0}{1}'.format(PATH, img_path), int(label_idx)))
                    else:
                        imgs.append('{0}{1}'.format(PATH, line))

                line = f.readline().strip('\n')
                 
        self.imags = imgs
        self.length = len(imgs)
        self.loader = loader
        self.transform = transform

    def __getitem__(self, mask):
        #print(self.imags[mask])
        if len(self.imags[mask]) == 2:
            data, label = self.imags[mask]
            data = self.loader(data)
            #if data.shape[0] < 100:
                
            #    in_transform = transforms.Compose([
            #        transforms.ToPILImage(),
            #        transforms.RandomCrop(299, padding = 250, padding_mode = 'reflect'),
            #        transforms.ToTensor(),
            #    ])
            #    data = in_transform(data)
            
            data = self.transform(data)
            return data, label
        else:
            return self.transform(self.imags[mask])

    def __len__(self):
        return self.length
'''
class MyDataSet(Dataset):
    def __init__(self, txt, PATH = 'C:/Users/admin/Desktop/Celeb-DF-v2-face/', loader = linear_loader, transform = None):
        super(MyDataSet, self).__init__()

        # txt = '{0}{1}'.format(path, txt)
        imgs = []
        #print(txt.split('/')[2].split('.')[0])

        with open(txt, 'r') as f:
            line = f.readline().strip('\n')
            while line:
                if line.strip() != '':
                    if ' ' in line:
                        label_idx, img_path = line.split(' ')
                        #print('{0}{1}'.format(PATH, img_path))
                        
                        #celeb
                        img_file = '{0}{1}'.format(PATH, img_path)
                        if os.path.exists(img_file):
                            pic_list = os.listdir(img_file)
                        else:
                            continue

                        for i, pic in enumerate(pic_list):
                            #print('{0}/{1}'.format(img_file, pic))
                            imgs.append(('{0}/{1}'.format(img_file, pic), int(label_idx)))
                            
                        #imgs.append(('{0}{1}'.format(PATH, img_path), int(label_idx)))
                    else:
                        imgs.append('{0}{1}'.format(PATH, line))

                line = f.readline().strip('\n')
                 
        self.imags = imgs
        self.length = len(imgs)
        self.loader = loader
        self.transform = transform

    def __getitem__(self, mask):
        #print(self.imags[mask])
        if len(self.imags[mask]) == 2:
            data, label = self.imags[mask]
            data = self.loader(data)
            #if data.shape[0] < 100:
                
            #    in_transform = transforms.Compose([
            #        transforms.ToPILImage(),
            #        transforms.RandomCrop(299, padding = 250, padding_mode = 'reflect'),
            #        transforms.ToTensor(),
            #    ])
            #    data = in_transform(data)
            
            data = self.transform(data)
            return data, label
        else:
            return self.transform(self.imags[mask])

    def __len__(self):
        return self.length

class ExpDataSet(Dataset):
    def __init__(self, txt, loader = linear_loader, transform = None):
        super(ExpDataSet, self).__init__()

        # txt = '{0}{1}'.format(path, txt)
        imgs = []
        with open(txt, 'r') as f:
            line = f.readline().strip('\n')
            while line:
                if line.strip() != '':
                    if ' ' in line:
                        line, _ = line.split(' ')
                    if line != '':
                        img_path = '{0}{1}'.format(path, line)
                                                                                            
                        try:
                            imgs.append(imread(img_path))
                        except Exception as e:
                            print(e)
                            imgs.append(imread('../val/0004/4617.png'))
                    line = f.readline().strip('\n')
        self.imags = imgs
        self.length = len(imgs)
        self.loader = loader
        self.transform = transform
                                                                                                                                                                                                                                                                                                                                
    def __getitem__(self, mask):
        return self.transform(self.imags[mask])

    def __len__(self):
        return self.length

def write_txt(data, txt):
    with open(txt, 'a') as f:
        print(data)
        f.writelines(data)
        f.write('\n')

if __name__=='__main__':
    start = time.time()
    train_data = DFDataset(txt = 'datasets/Celeb-DF-v2/train_txt.txt', transform = transforms.ToTensor())
    print(time.time() - start)
    #for i in range(0, train_data.__len__()):
    #    write_txt(train_data.__getitem__(i), 'datasets/Celeb-DF-v2/loadtxt_all.txt')
    #print(train_data.__getitem__(0))
    #print(train_data.__len__())


