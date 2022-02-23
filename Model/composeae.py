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
class ComplexProjectionModule(torch.nn.Module):
    def __init__(self, image_embed_dim=2048, text_embed_dim=2048):
        super(ComplexProjectionModule,self).__init__()
        self.bert_features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(text_embed_dim),
            torch.nn.Linear(text_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim)
        )
        self.image_features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
        )

    def forward(self, x):
        x1 = self.image_features(x[0])
        x2 = self.bert_features(x[1])
        # default value of CONJUGATE is 1. Only for rotationally symmetric loss value is -1.
        # which results in the CONJUGATE of text features in the complex space
        CONJUGATE = x[2]
        num_samples = x[0].shape[0]
        CONJUGATE = CONJUGATE[:num_samples]
        delta = x2  # text as rotation
        re_delta = torch.cos(delta)
        im_delta = CONJUGATE * torch.sin(delta)

        re_score = x1 * re_delta
        im_score = x1 * im_delta

        concat_x = torch.cat([re_score, im_score], 1)
        x0copy = x[0].unsqueeze(1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        re_score = re_score.unsqueeze(1)
        im_score = im_score.unsqueeze(1)

        return concat_x, x1, x2, x0copy, re_score, im_score

class LinearMapping(torch.nn.Module):
    """
    This is linear mapping to image space. rho(.)
    """

    def __init__(self, image_embed_dim=2048):
        super(LinearMapping,self).__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim)
        )

    def forward(self, x):
        theta_linear = self.mapping(x[0])
        return theta_linear
class ConvMapping(torch.nn.Module):
    """
    This is convoultional mapping to image space. rho_conv(.)
    """

    def __init__(self, image_embed_dim=2048):
        super(ConvMapping,self).__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim)
        )
        # in_channels, output channels
        self.conv = torch.nn.Conv1d(5, 64, kernel_size=3, padding=1)
        self.adaptivepooling = torch.nn.AdaptiveMaxPool1d(64)

    def forward(self, x):
        concat_features = torch.cat(x[1:], 1)
        concat_x = self.conv(concat_features)
        concat_x = self.adaptivepooling(concat_x)
        final_vec = concat_x.reshape((concat_x.shape[0], 4096))
        theta_conv = self.mapping(final_vec)
        return theta_conv


class ComposeAE(ImageEncoderTextEncoderBase):

    def __init__(self, args, **kwargs):
        super(ComposeAE, self).__init__(**kwargs)
        self.args = args
        self.texts = kwargs.get('texts')
        self.text_method = kwargs.get('text_method')
        self.w = nn.Parameter(torch.FloatTensor([1.0, 10.0, 1.0, 1.0]))
        self.linear_layer = nn.Linear(self.out_feature_image,4262)
        self.encoderLinear = torch.nn.Sequential(
            ComplexProjectionModule(),
            LinearMapping()
        )
        self.encoderWithConv = torch.nn.Sequential(
            ComplexProjectionModule(),
            ConvMapping()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.out_feature_image),
            torch.nn.ReLU(),
            torch.nn.Linear(self.out_feature_image, self.out_feature_image),
            torch.nn.ReLU(),
            torch.nn.Linear(self.out_feature_image, self.out_feature_image)
        )
        self.txtdecoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(self.out_feature_image),
            torch.nn.ReLU(),
            torch.nn.Linear(self.out_feature_image, self.out_feature_image),
            torch.nn.ReLU(),
            torch.nn.Linear(self.out_feature_image, self.out_feature_image)
        )
        self.model = nn.ModuleDict(self.model)

        # optimizer
        self.opt = torch.optim.AdamW(
            self.get_config_optim(args.lr),
            lr=args.lr,
            betas=(0.55, 0.999)
        )

    def compose_img_text(self, imgs, texts):
        image_features = self.extract_image_feature(imgs)
        text_features = self.extract_text_feature(texts)
        return self.compose_image_text_features(image_features, text_features)

    def compose_image_text_features(self, image_features, text_features, CONJUGATE = Variable(torch.cuda.FloatTensor(64, 1).fill_(1.0), requires_grad=False)):
        theta_linear = self.encoderLinear((image_features, text_features, CONJUGATE))
        theta_conv = self.encoderWithConv((image_features, text_features, CONJUGATE))
        theta = theta_linear * self.w[1] + theta_conv * self.w[0]
        return theta

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
        loss = final_loss(predict, tag)
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
            x_f = self.compose_img_text(x[0][0], x[1][1])
        else:
            x_f = self.compose_img_text(x[0][0], x[1][1])

        #x_f = self.model['final_layer'](x_f)
        x_f = self.linear_layer(x_f)
        x_t = x[2][1]
        return (x_f, x_t)

