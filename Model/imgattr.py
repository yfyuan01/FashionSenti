import math
import string
import random
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as M
from torch.autograd import Variable
import Model.resnet as resnet
from Model.fusion import ConCatModule
from Model.base import ImageEncoderTextEncoderBase
class NormalizationLayer(torch.nn.Module):
    def __init__(self,normalize_scale=1.0,learn_scale=True):
        super(NormalizationLayer,self).__init__()
        self.norm_s = float(normalize_scale)
        if learn_scale:
            self.norm_s = torch.nn.Parameter(torch.FloatTensor((self.norm_s,)))

        def forward(self, x):
            features = self.norm_s * x/torch.norm(x,dim=1,keepdim=True).expand_as(x)
            return features

class ImgAttr(ImageEncoderTextEncoderBase):
    """The TIRG model.
    The method is described in
    Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.
    "Composing Text and Image for Image Retrieval - An Empirical Odyssey"
    CVPR 2019. arXiv:1812.07119
    """

    def __init__(self, args, **kwargs):
        super(ImgAttr, self).__init__(**kwargs)

        self.args = args
        self.texts = kwargs.get('texts')
        self.text_method = kwargs.get('text_method')
        normalize_scale = args.normalize_scale
        self.w = nn.Parameter(torch.FloatTensor([1.0, 10.0, 1.0, 1.0]))
        self.model['norm'] = NormalizationLayer(learn_scale=True,
                                                normalize_scale=normalize_scale)

        self.model['gated_feature_composer'] = torch.nn.Sequential(
            ConCatModule(),
            nn.BatchNorm1d(self.out_feature_image + self.out_feature_text),
            nn.ReLU(),
            nn.Linear(self.out_feature_image + self.out_feature_text, self.out_feature_image + self.out_feature_text),
            nn.BatchNorm1d(self.out_feature_image + self.out_feature_text),
            nn.ReLU(),
            nn.Linear(self.out_feature_image + self.out_feature_text, self.out_feature_image),
        )

        self.model['res_info_composer'] = torch.nn.Sequential(
            ConCatModule(),
            nn.BatchNorm1d(self.out_feature_image + self.out_feature_text),
            nn.ReLU(),
            nn.Linear(self.out_feature_image + self.out_feature_text, self.out_feature_image + self.out_feature_text),
            nn.ReLU(),
            nn.Linear(self.out_feature_image + self.out_feature_text, self.out_feature_image),
        )
        self.classnum = 3
        self.model['linear'] = nn.Linear(self.out_feature_image, self.classnum)
        self.model['attr'] = nn.Linear(900,self.out_feature_image)
        self.model = nn.ModuleDict(self.model)
        # optimizer
        self.opt = torch.optim.AdamW(
            self.get_config_optim(args.lr),
            lr=args.lr,
            betas=(0.55, 0.999)
        )

    def compose_img_attr(self, imgs, texts):
        image_features = self.extract_image_feature(imgs)
        text_features = self.extract_tag_attribute_feature(texts).mean(dim=1)
        text_features = self.model['attr'](text_features)
        print text_features.size
        return self.compose_image_text_features(image_features, text_features)

    def compose_image_text_features(self, image_features, text_features):
        f1 = self.model['gated_feature_composer']((image_features, text_features))
        f2 = self.model['res_info_composer']((image_features, text_features))
        f = torch.sigmoid(f1) * image_features * self.w[0] + f2 * self.w[1]
        return f

    def get_config_optim(self, lr):
        params = []
        for k, v in self.model.items():
            if k == 'backbone':
                params.append({'params': v.parameters(), 'lr': lr, 'lrp': float(self.args.lrp)})
            else:
                params.append({'params': v.parameters(), 'lr': lr, 'lrp': 1.0})
        return params

    def adjust_lr(self, lr):
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr * param_group['lrp']

    def save(self, path, state={}):
        state['state_dict'] = dict()
        for k, v in self.model.items():
            state['state_dict'][k] = v.state_dict()
        state['texts'] = self.texts
        torch.save(state, path)

    def load(self, path):
        state_dict = torch.load(path)['state_dict']
        for k, v in state_dict.items():
            self.model[k].load_state_dict(v)

    def get_original_image_feature(self, x):
        '''
        x = image
        '''
        x = self.extract_image_feature(x)
        return self.model['norm'](x)

    def get_manipulated_image_feature(self, x):
        '''
        x[0] = (x_c, c_c, data['c_id'])
        x[1] = (we, w_key, text)
        '''
        if self.text_method == 'swem':
            x = self.compose_img_text(x[0][0], x[2][2])
        else:
            x = self.compose_img_text(x[0][0], x[2][2])
        return self.model['norm'](x)

    def update(self, predict, tag):
        '''
        input = (
            (img, gid, data['c_id']),
            (we_key, text),
            (name,tag)
        )
        '''
        # loss
        final_loss = nn.CrossEntropyLoss()
        loss = final_loss(predict,tag)
        # backward
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # return log
        log_data = dict()
        log_data['loss'] = float(loss.data)
        return log_data


    def forward(self, x):
        if self.text_method == 'swem':
            x_f = self.compose_img_attr(x[0][0], x[2][2])
        else:
            x_f = self.compose_img_attr(x[0][0], x[2][2])
        #x_f = self.model['final_layer'](x_f)
        x_f = self.model['linear'](x_f)
        x_t = x[2][1]
        return (x_f, x_t)
