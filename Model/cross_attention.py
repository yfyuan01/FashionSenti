import torch
import torchvision
import torch.nn as nn
import argparse
from base import ImageEncoderTextEncoderBase
from Preprocess.loss import NormalizationLayer
class CrossAttentionModule(nn.Module):
    def __init__(self,args,**kwargs):
        super(CrossAttentionModule, self).__init__(**kwargs)
        self.args = args
        self.in_feature_text = 3*300
        self.linear_layer1 = nn.Sequential(
            nn.Linear(2*self.in_feature_text,self.in_feature_text),
            nn.Tanh()
        )
        self.linear_layer2 = nn.Sequential(
            nn.Linear(2 * self.in_feature_text,self.in_feature_text),
            nn.Tanh())
    # x1 is the output of image-text turns : batch_size * max_turn_len * 2048
    # x2 is the output of target image: batch_size * feature_size * 2048
    def A_To_Q(self, x1, x2):
        h_q = list(x1.size())[1]
        h_a = list(x2.size())[1]
        e = list(x1.size())[2]
        reshape_q = x1.unsqueeze(2)
        reshape_a = x2.unsqueeze(1)
        reshape_q = reshape_q.repeat([1,1,h_a,1])
        reshape_a = reshape_a.repeat([1,h_q,1,1])
        combine = torch.cat([reshape_q,reshape_a],dim=3).reshape([-1,2*e]) # (batch_size*max_turn_len*feature_size)*2e
        M = self.linear_layer1(combine).reshape([-1,h_q,h_a,e]) #(batch_size*max_turn_len*feature_size)*e
        S = torch.softmax(M,dim=1)
        attentive_q = (S*reshape_q).sum(dim=1) #batch_size*feature_size*e
        similarity = (attentive_q*x2) #batch_size*feature_size
        # .sum(dim=2)
        # return attentive_q
        return similarity

    def Q_To_A(self, x1, x2):
        similarity = self.A_To_Q(x1,x2)
        h_a = list(x2.size())[1]
        e = list(x2.size())[2]
        avg_q = torch.mean(x1,dim=1) #batch_size*e
        reshape_q = avg_q.unsqueeze(1)
        reshape_q = reshape_q.repeat([1,h_a,1]) #batch_size*feature_size*e
        combine = torch.cat([reshape_q,x2],dim=2).reshape([-1,2*e])
        M = self.linear_layer2(combine).reshape([-1,h_a,e]) #batch_size*feature_size
        S = torch.softmax(M,dim=1)
        attentive_a = (S*similarity).sum(dim=1)
        return attentive_a
class Combine(ImageEncoderTextEncoderBase):
    def __init__(self,args,**kwargs):
        super(Combine, self).__init__(**kwargs)
        self.args = args
        self.texts = kwargs.get('texts')
        self.cross_attention = CrossAttentionModule(args=self.args)
        self.model['norm'] = NormalizationLayer(learn_scale=True,
                                                normalize_scale=self.args.normalize_scale)
        self.classnum = 3
        self.text_dim = 3*300
        self.linear_layer = nn.Linear(self.text_dim*2,self.classnum)
        self.model['softmax'] = nn.Softmax()
        self.model = nn.ModuleDict(self.model)
        self.opt = torch.optim.AdamW(
        self.get_config_optim(args.lr),
        lr = args.lr,
        betas = (0.55,0.999)
        )
    def update(self, predict, tag):
        final_loss = nn.CrossEntropyLoss()
        loss = final_loss(predict, tag)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # return log
        log_data = dict()
        log_data['loss'] = float(loss.data)
        return log_data
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
    def get_original_image_feature(self, x):
        x = self.extract_image_feature(x)
        return self.model['norm'](x)
    def get_original_tag_feature(self,x):
        x = self.extract_tag_feature(x).mean(1)
        return self.model['norm'](x)
    def get_original_combined_feature(self,x,y):
        x = self.extract_tag_feature(x)
        y = self.extract_image_feature(y).unsqueeze(1)
        z = torch.cat((x,y),1).mean(1)
        return self.model['norm'](z)
    def save(self, path, state={}):
        state['state_dict'] = dict()
        for k,v in self.model.items():
            state['state_dict'][k] = v.state_dict()
        state['texts'] = self.texts
        torch.save(state,path)
    def load(self, path):
        state_dict = torch.load(path)['state_dict']
        for k,v in state_dict.items():
            self.model[k].load_state_dict(v)
    def forward(self, x):
        x_txt = self.extract_tag_text_feature(x[1][1])
        x_tag = self.extract_tag_attribute_feature(x[2][2])
        #x_1 = self.cross_attention.A_To_Q(x_txt,x_tag)
        #print 'x_1'
        print x_1.size()
        x_2 = self.cross_attention.Q_To_A(x_txt,x_tag)
        #print 'x_2'
        print x_2.size()
        # x_1 = self.model['norm'](x_1)
        # x_2 = self.model['norm'](x_2)
        x_f = self.linear_layer(x_2)
        #x_f = self.linear_layer(torch.cat((x_1.mean(dim=1),x_2.mean(dim=1)),dim=1))
        x_t = x[2][1]
        return (x_f,x_t)







def main():
    x1 = torch.randn(3,4,10)
    x2 = torch.randn(3,2,10)
    Co = Combine(args=args)
    CrossAttention = CrossAttentionModule(args=args)
    print CrossAttention.Q_To_A(x1,x2).data
    print Co.cal_attention(x1,x1,x1).size()
if __name__ == '__main__':
    parser = argparse.ArgumentParser('A Test of this Module')
    parser.add_argument('--fdims',default='10',type=int)
    args, _ = parser.parse_known_args()
    main()



