import torch
import torch.nn as nn
from Model.base import ImageEncoderTextEncoderBase

class TextOnlyModel(ImageEncoderTextEncoderBase):
    def __init__(self,args,**kwargs):
        super(TextOnlyModel,self).__init__(**kwargs)
        self.args = args
        self.texts = kwargs.get('texts')
        self.text_method = kwargs.get('text_method')
        normalize_scale = args.normalize_scale
        self.model = nn.ModuleDict(self.model)
        self.opt = torch.optim.AdamW(
            self.get_config_optim(args.lr),
            lr = args.lr,
            betas = (0.55,0.999)
        )
        self.classnum = 4262
        self.linear = nn.Linear(self.args.fdims, self.classnum)
    def get_config_optim(self,lr):
        params = []
        for k,v in self.model.items():
            if k=='backbone':
                params.append({'params':v.parameters(),'lr':lr*0.1})
            else:
                params.append({'params':v.parameters(),'lr':lr})
        return params

    def adjust_lr(self,lr):
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr

    def save(self,path,state={}):
        state['state_dict'] = dict()
        for k,v in self.model.items():
            state['state_dict'][k] = v.state_dict()
        state['texts'] = self.texts
        torch.save(state,path)

    def load(self,path):
        state_dict = torch.load(path)['state_dict']
        for k,v in state_dict.items():
            self.model[k].load_state_dict(v)

    def update(self, predict, tag):
        final_loss = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        loss = final_loss(predict, tag)
        # backward
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        log_data = dict()
        log_data['loss'] = float(loss.data)
        return log_data

    def forward(self, x):
        if self.text_method == 'swem':
            x_f = self.extract_text_feature(x[2][0])
        else:
            x_f = self.extract_text_feature(x[1][1])
        x_f = self.linear(x_f)
        x_t = x[2][1]
        return (x_f,x_t)
