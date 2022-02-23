#-*- coding:utf-8 -*- 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#from __future__ import unicode_literals
import argparse
import os
import sys
sys.path.append('../')
import random
import numpy as np
import json
import pickle
from tqdm import tqdm
from PIL import Image
# from Model.match import MatchTextOnly
import torch
import torch.utils.data as data
import torchvision.transforms as T
from Preprocess.transform import PaddedResize
import re
# data_root is the root of the dataset
def filter_emoji(text):
    try:
        text = unicode(text,"utf-8")
    except TypeError as e:
        pass
    try:
        highpoints = re.compile(u'[\U00010000-\U0010ffff]')
    except re.error:
        highpoints = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    return highpoints.sub(u'',text)
class TWMMS1(data.Dataset):
    def __init__(self,data_root,image_root,image_size=224,split='val'):
        self.data_root = data_root
        self.image_root = image_root
        self.image_size = image_size
        self.split = split
        self.transform = None
        self.reload()
    def __set_transform(self):
        IMAGE_SIZE = self.image_size
        if self.split == 'train':
            self.transform = T.Compose([
                PaddedResize(IMAGE_SIZE),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(degrees=45,translate=(0.15,0.15),scale=(0.9,1.1)),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
            ])
        elif self.split in ['test','val']:
            self.transform = T.Compose([
                PaddedResize(IMAGE_SIZE),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
            ])
    def __load_pil_image__(self,path):
        try:
            with open(path,'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except Exception as err:
            print(err)
            img = Image.new('RGB',(224,224))
            return img

    def __crop_image__(self,img,bbox):
        # left, top, right, buttom
        w, h = img.size
        x_min = int(w * bbox[0])
        y_min = int(h * bbox[1])
        x_max = x_min + int(w * bbox[2])
        y_max = x_max + int(h * bbox[3])
        crop_img = img.crop((x_min,y_min,x_max,y_max))
        return crop_img
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        return self.__sample__(index)

    def get_all_texts(self):
        return self.all_texts

    def get_all_attribute_texts(self):
        return self.all_attr_texts
    def __print_status(self):
        print("===============")
        print('Data Statistics: ')
        # print('{} Data Size: {}'.format(self.split,len(self.dataset)))
        print("===============")

    def __load_data__(self):
        raise NotImplementedError()
    def __sample__(self,index):
        raise NotImplementedError()
    def reload(self):
        self.__set_transform()
        self.__load_data__()
        self.__print_status()
    def get_loader(self, **kwargs):
        batch_size = kwargs.get('batch_size',16)
        num_workers = kwargs.get('workers',20)
        shuffle = False
        drop_last = False
        if self.split == 'train':
            shuffle = True
            drop_last = True
        data_loader = torch.utils.data.DataLoader(dataset=self,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  drop_last=drop_last)
        return data_loader
class TWMMS1TrainTestDataset(TWMMS1):
    def __init__(self, **kwargs):
        super(TWMMS1TrainTestDataset, self).__init__(**kwargs)
    def __load_data__(self):
        print('[Dataset] load caption annotations: {}'.format(self.data_root))
        self.dataset = []
        self.all_texts = []
        self.cls2idx = dict()
        self.idx2cls = list()
        with open(os.path.join(self.data_root,'final_dict_'+self.split+'.pkl'),'rb') as f:
            self.data_dict = pickle.load(f)
        self.all_attr_texts = []
        print('Finished Caption Loading')
        print('[Dataset] load attribute annotations: {}'.format(self.data_root))
        with open(os.path.join(self.data_root,'attribute_dict.pkl'),'rb') as f:
            self.attr_dict = pickle.load(f)
        for i,v in enumerate(self.attr_dict.values()):
            self.all_attr_texts.extend(v.split(' '))
        self.all_attr_texts = list(set(self.all_attr_texts))
        print('Finished Attributes Loading')
        with open(os.path.join(self.data_root,'vocab.pkl'),'rb') as f:
            num_dict = pickle.load(f)
        # with open(os.path.join(self.data_root,'imgfeature/face_dict.pkl'),'rb') as f:
        #     self.face_dict = pickle.load(f)
        # with open(os.path.join(self.data_root,'imgfeature/item_dict.pkl'), 'rb') as f:
        #     self.item_dict = pickle.load(f)
        print('Finished Other Information Loading')
        for i,d in enumerate(tqdm(self.data_dict.values())):
            id = list(self.data_dict.keys())[i]
            # num_dict = {'Positive': 2, 'Negative': 0, 'Neutral': 1}
            if d['tags'].split(';')[0] in num_dict:
                tag = num_dict[d['tags'].split(';')[0]]
            else:
                tag = num_dict['<unk>']
            captions = filter_emoji(d['text'])
            caption_list = captions.split(' ')
            if not id in self.cls2idx:
                self.cls2idx[id] = len(self.cls2idx)
                self.idx2cls.append(id)
            self.all_texts.extend(caption_list)
            text = [x.strip() for x in caption_list]
            attr = self.attr_dict[id].split(' ')
            #attr = attr_record.values()
            # face = self.face_dict[id]
            # if (self.item_dict[id]!=[]):
            #     box = self.item_dict[id][0][2:]
            # else:
            #     box = []
            random.shuffle(text)
            text = '[CLS]' + ' '.join(text)
            we_key = '{}_{}_{}'.format(self.split, id, i)
            _data = {
                    'img_path': os.path.join(self.image_root, '{}'.format(id)),
                    'id': id,
                    'tag': tag,
                    'we_key': we_key,
                    'text': text,
                    'attr': attr,
                }
            self.dataset.append(_data)
        self.dataset = np.asarray(self.dataset)

    def __sample__(self,index):
        data = self.dataset[index]
        img = self.__load_pil_image__(data['img_path'])
        # face = data['face']
        # box = data['box']
        gid = self.cls2idx[data['id']]
        we_key = data['we_key']
        tag = data['tag']
        attr = data['attr']
        if not self.transform is None:
            img = self.transform(img)
            # img_list = [self.transform(image) for image in img_list]
        # print("this is a test")
        return (
            (img,gid,data['id']),
            (data['we_key'],data['text']),
            (we_key,tag,attr)
        )

class MyDataset(data.Dataset):
    def __init__(self,data_root,image_root,image_size=224,split='val'):
        self.data_root = data_root
        self.image_root = image_root
        self.image_size = image_size
        self.split = split
        self.transform = None
        self.reload()
    def __set_transform(self):
        IMAGE_SIZE = self.image_size
        if self.split == 'train':
            self.transform = T.Compose([
                PaddedResize(IMAGE_SIZE),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(degrees=45,translate=(0.15,0.15),scale=(0.9,1.1)),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
            ])
        elif self.split in ['test','val']:
            self.transform = T.Compose([
                PaddedResize(IMAGE_SIZE),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
            ])
    def __load_pil_image__(self,path):
        try:
            with open(path,'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except Exception as err:
            print(err)
            img = Image.new('RGB',(224,224))
            return img

    def __crop_image__(self,img,bbox):
        # left, top, right, buttom
        w, h = img.size
        x_min = int(w * bbox[0])
        y_min = int(h * bbox[1])
        x_max = x_min + int(w * bbox[2])
        y_max = x_max + int(h * bbox[3])
        crop_img = img.crop((x_min,y_min,x_max,y_max))
        return crop_img
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        return self.__sample__(index)

    def get_all_texts(self):
        return self.all_texts

    def get_all_attribute_texts(self):
        return self.all_attr_texts
    def __print_status(self):
        print("===============")
        print('Data Statistics: ')
        # print('{} Data Size: {}'.format(self.split,len(self.dataset)))
        print("===============")

    def __load_data__(self):
        raise NotImplementedError()
    def __sample__(self,index):
        raise NotImplementedError()
    def reload(self):
        self.__set_transform()
        self.__load_data__()
        self.__print_status()
    def get_loader(self, **kwargs):
        batch_size = kwargs.get('batch_size',16)
        num_workers = kwargs.get('workers',20)
        shuffle = False
        drop_last = False
        if self.split == 'train':
            shuffle = True
            drop_last = True
        data_loader = torch.utils.data.DataLoader(dataset=self,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  drop_last=drop_last)
        return data_loader
class MyTrainTestDataset(MyDataset):
    def __init__(self, **kwargs):
        super(MyTrainTestDataset, self).__init__(**kwargs)
    def __load_data__(self):
        print('[Dataset] load caption annotations: {}'.format(self.data_root))
        self.dataset = []
        self.all_texts = []
        self.cls2idx = dict()
        self.idx2cls = list()
        with open(os.path.join(self.data_root,'final_dict_'+self.split+'.pkl'),'rb') as f:
            self.data_dict = pickle.load(f)
        self.all_attr_texts = []
        print('Finished Caption Loading')
        print('[Dataset] load attribute annotations: {}'.format(self.data_root))
        with open(os.path.join(self.data_root,'attribute_dict1.pkl'),'rb') as f:
            self.attr_dict = pickle.load(f)
        for i,v in enumerate(self.attr_dict.values()):
            for vv in v.values():
                self.all_attr_texts.extend(vv.split(' '))
        self.all_attr_texts = list(set(self.all_attr_texts))
        print('Finished Attributes Loading')
        with open(os.path.join(self.data_root,'imgfeature/face_dict.pkl'),'rb') as f:
            self.face_dict = pickle.load(f)
        with open(os.path.join(self.data_root,'imgfeature/item_dict.pkl'), 'rb') as f:
            self.item_dict = pickle.load(f)
        print('Finished Other Information Loading')
        for i,d in enumerate(tqdm(self.data_dict.values())):
            id = d['img_id']
            num_dict = {'Positive': 2, 'Negative': 0, 'Neutral': 1}
            tag = num_dict[d['sentiment']]
            captions = filter_emoji(d['text'])
            caption_list = captions.split(' ')
            if not id in self.cls2idx:
                self.cls2idx[id] = len(self.cls2idx)
                self.idx2cls.append(id)
            self.all_texts.extend(caption_list)
            text = [x.strip() for x in caption_list]
            attr_record = self.attr_dict[id]
            attr = attr_record.values()
            face = self.face_dict[id]
            if (self.item_dict[id]!=[]):
                box = self.item_dict[id][0][2:]
            else:
                box = []
            random.shuffle(text)
            text = '[CLS]' + ' '.join(text)
            we_key = '{}_{}_{}'.format(self.split, id, i)
            _data = {
                    'img_path': os.path.join(self.image_root, '{}.jpg'.format(id)),
                    'id': id,
                    'tag': tag,
                    'we_key': we_key,
                    'text': text,
                    'attr': attr,
                    'face': face,
                    'box': box
                }
            self.dataset.append(_data)
        self.dataset = np.asarray(self.dataset)

    def __sample__(self,index):
        data = self.dataset[index]
        img = self.__load_pil_image__(data['img_path'])
        face = data['face']
        box = data['box']
        gid = self.cls2idx[data['id']]
        we_key = data['we_key']
        tag = data['tag']
        attr = data['attr']
        if box!=[]:
            cropped = img.crop(tuple([box[0],box[2],box[1],box[3]]))
        else:
            cropped = img
        if not self.transform is None:
            img = self.transform(img)
            cropped = self.transform(cropped)
            # img_list = [self.transform(image) for image in img_list]
        # print("this is a test")
        return (
            (img,gid,data['id'],face,cropped),
            (data['we_key'],data['text']),
            (we_key,tag,attr)
        )

class Bt4saDataset(data.Dataset):
    def __init__(self,data_root,image_size=224,split='val'):
        self.data_root = data_root
        self.tag_root = os.path.join(self.data_root,'t4sa_text_sentiment.tsv')
        self.cap_root = os.path.join(self.data_root,'raw_tweets_text.csv')
        self.image_size = image_size
        self.split = split
        self.transform = None
        self.reload()
    def __set_transform(self):
        IMAGE_SIZE = self.image_size
        if self.split == 'train':
            self.transform = T.Compose([
                PaddedResize(IMAGE_SIZE),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(degrees=45,translate=(0.15,0.15),scale=(0.9,1.1)),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
            ])
        elif self.split in ['test','val']:
            self.transform = T.Compose([
                PaddedResize(IMAGE_SIZE),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
            ])
    def __load_pil_image__(self,path):
        try:
            with open(path,'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except Exception as err:
            print(err)
            img = Image.new('RGB',(224,224))
            return img

    def __crop_image__(self,img,bbox):
        # left, top, right, buttom
        w, h = img.size
        x_min = int(w * bbox[0])
        y_min = int(h * bbox[1])
        x_max = x_min + int(w * bbox[2])
        y_max = x_max + int(h * bbox[3])
        crop_img = img.crop((x_min,y_min,x_max,y_max))
        return crop_img
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.__sample__(index)

    def get_all_texts(self):
        return self.all_texts

    def __print_status(self):
        print("===============")
        print('Data Statistics: ')
        # print('{} Data Size: {}'.format(self.split,len(self.dataset)))
        print("===============")

    def __load_data__(self):
        raise NotImplementedError()
    def __sample__(self,index):
        raise NotImplementedError()
    def reload(self):
        self.__set_transform()
        self.__load_data__()
        self.__print_status()
    def get_loader(self, **kwargs):
        batch_size = kwargs.get('batch_size',16)
        num_workers = kwargs.get('workers',20)
        shuffle = False
        drop_last = False
        if self.split == 'train':
            shuffle = True
            drop_last = True

        data_loader = torch.utils.data.DataLoader(dataset=self,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  drop_last=drop_last)
        return data_loader
class Bt4saTrainTestDataset(Bt4saDataset):
    def __init__(self, **kwargs):
        super(Bt4saTrainTestDataset, self).__init__(**kwargs)
    def __load_tag(self):
        print('Loading bt4sa tag file....')
        with open(self.tag_root,'r') as f:
            tag_list = f.readlines()
        tag_list = [t.strip('\n').split('\t') for t in tag_list][1:]
        tag_list = [[t[0],float(t[1]),float(t[2]),float(t[3])] for t in tag_list]
        tag_dict = {t[0]:t.index(max(t[1:]))-1 for t in tag_list}
        if not os.path.exists(os.path.join(self.data_root,'tag.pkl')):
            with open(os.path.join(self.data_root,'tag.pkl'),'w') as f:
                print('Writing tag file...')
                pickle.dump(tag_dict,f)
        return tag_dict
    def __load_cap(self):
        print('Loading bt4sa cap file....')
        with open(self.cap_root,'r') as f:
            cap_list = f.readlines()
        cap_list = [c.strip('\n').split(',') for c in cap_list]
        cap_dict = {c[0]:','.join(c[1:]) for c in cap_list}
        cap_dict = {c.lstrip('"').rstrip('"'):re.sub(r"http\S+", "", cap_dict[c]) for c in cap_dict.keys()}
        if not os.path.exists(os.path.join(self.data_root,'cap.pkl')):
            with open(os.path.join(self.data_root,'cap.pkl'),'w') as f:
                print('Writing cap file...')
                pickle.dump(cap_dict,f)
        return cap_dict
    def __load_data__(self):
        print('[Dataset] load caption annotations: {}'.format(self.data_root))
        self.dataset = []
        self.all_texts = []
        self.cls2idx = dict()
        self.idx2cls = list()
        if self.split == 'train':
            self.tag_dict = self.__load_tag()
            self.cap_dict = self.__load_cap()
        else:
            print('Recovering files...')
            with open(os.path.join(self.data_root,'tag.pkl'),'r') as f:
                self.tag_dict = pickle.load(f)
            with open(os.path.join(self.data_root,'cap.pkl'),'r') as f:
                self.cap_dict = pickle.load(f)
        print('Finished Loading')
        all_root = os.path.join(self.data_root,self.split)
        file_list = os.listdir(all_root)
        for i,d in enumerate(tqdm(file_list)):
            id = d.split('-')[0]
            tag = self.tag_dict[id]
            name = d[:-4]
            captions = filter_emoji(self.cap_dict[id])
            caption_list = captions.split(' ')
            if not id in self.cls2idx:
                self.cls2idx[id] = len(self.cls2idx)
                self.idx2cls.append(id)
            self.all_texts.extend(caption_list)
            text = [x.strip() for x in caption_list]
            random.shuffle(text)
            text = '[CLS]' + ' [SEP] '.join(text)
            we_key = '{}_{}_{}'.format(self.split, id, i)
            _data = {
                    'img_path': os.path.join(self.data_root, self.split+'/{}.jpg'.format(name)),
                    'id': id,
                    'name': name,
                    'tag': tag,
                    'we_key': we_key,
                    'text': text
                }
            self.dataset.append(_data)
        self.dataset = np.asarray(self.dataset)

    def __sample__(self,index):
        data = self.dataset[index]
        img = self.__load_pil_image__(data['img_path'])
        gid = self.cls2idx[data['id']]
        we_key = data['we_key']
        name = data['name']
        tag = data['tag']
        if not self.transform is None:
            img = self.transform(img)
        # print("this is a test")
        return (
            (img,gid,data['id']),
            (data['we_key'],data['text']),
            (name,tag)
        )
class TwitterDataset(data.Dataset):
    def __init__(self,data_root,image_root,tag_root,
                 image_size=224,split='val'):
        self.data_root = data_root
        self.image_root = image_root
        self.tag_root = tag_root
        self.image_size = image_size
        self.split = split
        self.transform = None
        self.range = random.sample(range(1,6), 4)
        self.reload()
    # image data augmentation
    def __set_transform(self):
        IMAGE_SIZE = self.image_size
        if self.split == 'train':
            self.transform = T.Compose([
                PaddedResize(IMAGE_SIZE),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(degrees=45,translate=(0.15,0.15),scale=(0.9,1.1)),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
            ])
        elif self.split in ['test','val']:
            self.transform = T.Compose([
                PaddedResize(IMAGE_SIZE),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
            ])
    def __load_pil_image__(self,path):
        try:
            with open(path,'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except Exception as err:
            print(err)
            img = Image.new('RGB',(224,224))
            return img

    def __crop_image__(self,img,bbox):
        # left, top, right, buttom
        w, h = img.size
        x_min = int(w * bbox[0])
        y_min = int(h * bbox[1])
        x_max = x_min + int(w * bbox[2])
        y_max = x_max + int(h * bbox[3])
        crop_img = img.crop((x_min,y_min,x_max,y_max))
        return crop_img

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.__sample__(index)

    def get_all_texts(self):
        return self.all_texts

    def __print_status(self):
        print("===============")
        print('Data Statistics: ')
        # print('{} Data Size: {}'.format(self.split,len(self.dataset)))
        print("===============")

    def __load_data__(self):
        raise NotImplementedError()
    def __sample__(self,index):
        raise NotImplementedError()
    def reload(self):
        self.__set_transform()
        self.__load_data__()
        self.__print_status()
    def get_loader(self, **kwargs):
        batch_size = kwargs.get('batch_size',16)
        num_workers = kwargs.get('workers',20)
        shuffle = False
        drop_last = False
        if self.split == 'train':
            shuffle = True
            drop_last = True

        data_loader = torch.utils.data.DataLoader(dataset=self,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  drop_last=drop_last)
        return data_loader
class TwitterTrainValDataset(TwitterDataset):
    def __init__(self, **kwargs):
        super(TwitterTrainValDataset, self).__init__(**kwargs)
    def __load_tag(self):
        with open(self.tag_root,'r') as f:
            tag_list = f.readlines()
        tag_list = [t.strip('\r\n') for t in tag_list][1:]
        num_dict = {'positive': 2, 'negative': 0, 'neutral': 1}
        tag_dict = {t.split('\t')[0][:-4]:num_dict[t.split('\t')[1]] for t in tag_list}
        return tag_dict

    def __load_data__(self):
        # with open('../assets/image_embedding/embeddings_2.pkl','rb') as f:
        #     self.ie = pickle.load(f)
        print('[Dataset] load caption annotations: {}'.format(self.data_root))
        self.dataset = []
        self.all_texts = []
        self.cls2idx = dict()
        self.idx2cls = list()
        self.tag_dict = self.__load_tag()
        #for t in range(1,5):
        for t in self.range:
            all_root = os.path.join(self.data_root,'partition'+str(t))
            file_list = os.listdir(all_root)
            cap_file_list = [fi for fi in file_list if fi.find('.txt')>=0]
            for i,d in enumerate(tqdm(cap_file_list)):
                id = d.split('_')[0]
                name = d.split('_')[1][:-4]
                tag = self.tag_dict[id]
                with open(os.path.join(all_root,d),'r') as f:
                    caption = f.readlines()
                    caption_1 = []
                    for cc in caption:
                        try:
                            caption_1.append(filter_emoji(cc).strip('\n'))
                        except:
                            pass
                    caption = [cc for cc in caption_1 if cc!='']
                    captions = ' '.join(caption)
                    caption_list = captions.split(' ')
                if not id in self.cls2idx:
                    self.cls2idx[id] = len(self.cls2idx)
                    self.idx2cls.append(id)
                self.all_texts.extend(caption_list)
                text = [x.strip() for x in caption_list]
                random.shuffle(text)
                text = '[CLS]' + ' [SEP] '.join(text)
                we_key = '{}_{}_{}_{}'.format(self.split, t, id, i)
                _data = {
                    'img_path': os.path.join(self.image_root, 'partition'+str(t)+'/{}.jpg'.format(id)),
                    'id': id,
                    'name': name,
                    'tag': tag,
                    'we_key': we_key,
                    'text': text
                }
                self.dataset.append(_data)
        self.dataset = np.asarray(self.dataset)

    def __sample__(self,index):
        data = self.dataset[index]
        img = self.__load_pil_image__(data['img_path'])
        gid = self.cls2idx[data['id']]
        we_key = data['we_key']
        name = data['name']
        tag = data['tag']
        if not self.transform is None:
            img = self.transform(img)
        # print("this is a test")
        return (
            (img,gid,data['id']),
            (data['we_key'],data['text']),
            (name,tag)
        )
class TwitterTestDataset(TwitterDataset):
    def __init__(self, **kwargs):
        super(TwitterTestDataset, self).__init__(**kwargs)
    def __load_tag(self):
        with open(self.tag_root,'r') as f:
            tag_list = f.readlines()
        tag_list = [t.strip('\r\n') for t in tag_list][1:]
        num_dict = {'positive': 2, 'negative': 0, 'neutral': 1}
        tag_dict = {t.split('\t')[0][:-4]:num_dict[t.split('\t')[1]] for t in tag_list}
        return tag_dict

    def __load_data__(self):
        # with open('../assets/image_embedding/embeddings_2.pkl','rb') as f:
        #     self.ie = pickle.load(f)
        print('[Dataset] load caption annotations: {}'.format(self.data_root))
        self.dataset = []
        self.all_texts = []
        self.cls2idx = dict()
        self.idx2cls = list()
        self.tag_dict = self.__load_tag()
        t_l = [i for i in range(1,6) if i not in self.range]
        for t in t_l:
        #for t in range(5,6):
            all_root = os.path.join(self.data_root,'partition'+str(t))
            file_list = os.listdir(all_root)
            cap_file_list = [fi for fi in file_list if fi.find('.txt')>=0]
            for i,d in enumerate(tqdm(cap_file_list)):
                id = d.split('_')[0]
                name = d.split('_')[1][:-4]
                tag = self.tag_dict[id]
                with open(os.path.join(all_root,d),'r') as f:
                    caption = f.readlines()
                    caption_1 = []
                    for cc in caption:
                        try:
                            caption_1.append(filter_emoji(cc).strip('\n'))
                        except:
                            pass
                    caption = [cc for cc in caption_1 if cc!='']
                    captions = ' '.join(caption)
                    caption_list = captions.split(' ')
                if not id in self.cls2idx:
                    self.cls2idx[id] = len(self.cls2idx)
                    self.idx2cls.append(id)
                self.all_texts.extend(caption_list)
                text = [x.strip() for x in caption_list]
                random.shuffle(text)
                text = '[CLS]' + ' [SEP] '.join(text)
                we_key = '{}_{}_{}_{}'.format(self.split, t, id, i)
                _data = {
                    'img_path': os.path.join(self.image_root, 'partition'+str(t)+'/{}.jpg'.format(id)),
                    'id': id,
                    'name': name,
                    'tag': tag,
                    'we_key': we_key,
                    'text': text
                }
                self.dataset.append(_data)
        self.dataset = np.asarray(self.dataset)

    def __sample__(self,index):
        data = self.dataset[index]
        img = self.__load_pil_image__(data['img_path'])
        gid = self.cls2idx[data['id']]
        we_key = data['we_key']
        name = data['name']
        tag = data['tag']
        if not self.transform is None:
            img = self.transform(img)
        # print("this is a test")
        return (
            (img,gid,data['id']),
            (data['we_key'],data['text']),
            (name,tag)
        )

def main():
    if args.dataset=='bt4sa':
        train_dataset = Bt4saTrainTestDataset(
            data_root=args.data_root,
            image_size=args.image_size,
            split='train',
        )
        train_loader = train_dataset.get_loader(batch_size=args.batch_size)
        test_dataset = Bt4saTrainTestDataset(
            data_root=args.data_root,
            image_size=args.image_size,
            split='test',
        )
        test_loader = test_dataset.get_loader(batch_size=args.batch_size)
    elif args.dataset=='twitter':
        train_dataset = TwitterTrainValDataset(
            data_root=args.data_root,
            image_size=args.image_size,
            image_root=args.image_root,
            tag_root=args.tag_root,
            split='train',
        )
        train_loader = train_dataset.get_loader(batch_size=args.batch_size)
        test_dataset = TwitterTestDataset(
                data_root=args.data_root,
                image_size=args.image_size,
                image_root=args.image_root,
                tag_root=args.tag_root,
                split='val',
        )
        test_loader = test_dataset.get_loader(batch_size=args.batch_size)
    else:
        train_dataset = MyTrainTestDataset(
            data_root=args.data_root,
            image_root=args.image_root,
            image_size=args.image_size,
            split='train',
        )
        train_loader = train_dataset.get_loader(batch_size=args.batch_size)
        print(type(train_dataset.get_all_texts()))
        print(len(train_dataset.get_all_attribute_texts()[0]))
        test_dataset = MyTrainTestDataset(
            data_root=args.data_root,
            image_root=args.image_root,
            image_size=args.image_size,
            split='test',
        )
        test_loader = test_dataset.get_loader(batch_size=args.batch_size)
    print("start indexing")
    for bidx,input in enumerate(tqdm(test_loader,desc='Test')):
        print(input[0][0].size())
        print(len(input[1][1]))
        print(input[1][1])
        print(len(input[2][2]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('A Test of this Module')
    parser.add_argument('--data_root',required=False,type=str,default='../../')
    parser.add_argument('--image_size',default=224,type=int,help='image size (default: 16)')
    parser.add_argument('--batch_size',default=32,type=int)
    parser.add_argument('--text_method',default='lstm',choices=['lstm','swem','lstm-gru'],
                        type=str)
    parser.add_argument('--tag_root', default='../../data/Twitter_Dataset/twitter/groundTruthLabel.txt', type=str)
    parser.add_argument('--method',default='match-text-only',type=str,help='method')
    parser.add_argument('--backbone',default='resnet152',type=str)
    parser.add_argument('--image_root',default='/Users/yuanyifei/Documents/task202011/images',type=str)
    parser.add_argument('--dataset',default='mydataset',choices=['bt4sa','twitter','mydataset'])
    args, _ = parser.parse_known_args()
    main()
