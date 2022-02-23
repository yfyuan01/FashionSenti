import os
import json
import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from irbench.irbench import IRBench
from irbench.evals.eval_helper import EvalHelper
from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score
class Trainer(object):
    def __init__(self,
                 args,
                 data_loader,
                 model,
                 summary_writer):
        self.args = args
        self.data_loader = data_loader
        self.model = model
        self.summary_writer = summary_writer
        self.processed_images = 0
        self.global_step = 0

    def __adjust_lr__(self,epoch,warmup=True):
        lr = self.args.lr * self.args.batch_size/16.0
        if warmup:
            warmup_images = 10000
            lr = min(self.processed_images * lr / float(warmup_images),lr)
        for e in self.args.lr_decay_steps:
            if epoch>=e:
                lr*=self.args.lr_decay_steps
        self.model.adjust_lr(lr)
        self.cur_lr = lr

    def __logging__(self,log_data):
        msg = '[Train][{}]'.format(self.args.expr_name)
        msg += '[Epoch: {}]'.format(self.epoch)
        msg += '[Lr:{:.6f}]'.format(self.cur_lr)
        log_data['lr'] = self.cur_lr
        for k,v in log_data.items():
            if not self.summary_writer is None:
                self.summary_writer.add_scalar(k,v,self.global_step)
            if isinstance(v,float):
                msg += '{}:{:.6f}'.format(k,v)
            else:
                msg += '{}:{}'.format(k,v)
        print msg

    def train(self,epoch):
        self.epoch = epoch
        self.model.train()
        total = 0.
        correct = 0.
        for bidx, input in enumerate(tqdm(self.data_loader,desc='Train')):
            self.global_step+=1
            self.processed_images+=input[0][0].size(0)
            self.__adjust_lr__(epoch,warmup=self.args.warmup)
            input[0][0] = Variable(input[0][0].cuda())
            input[0][1] = Variable(input[0][1].cuda())
            #input[0][3] = Variable(input[0][3].cuda())
            #input[0][4] = Variable(input[0][4].cuda())
            input[2][1] = Variable(input[2][1].cuda())
            # input[3][1] = Variable(input[3][1].cuda())
            output = self.model(input)
            _, predicted = torch.max(output[0], 1)
            labels = output[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc = 100.0 * correct / total
            print('Accuracy of the network on the 10000 train images: %d %%' % (
                acc))
            log_data = self.model.update(output[0],output[1])
            #torch.save(self.model,'model.pkl')
            if (bidx % self.args.print_freq) == 0:
                self.__logging__(log_data)


class Evaluator(object):
    def __init__(self,
                 args,
                 data_loader,
                 model,
                 summary_writer,
                 eval_freq):
        self.args = args
        self.data_loader = data_loader
        self.model = model
        self.summary_writer = summary_writer
        self.eval_freq = eval_freq
        self.best_score = 0.
        self.repo_path = os.path.join('./repo',args.method)
        if not os.path.exists(self.repo_path):
            os.makedirs(self.repo_path)

    def test(self,epoch):
        self.epoch = epoch
        model = self.model.eval()
        total = 0.
        correct = 0.
        for bidx, input in enumerate(tqdm(self.data_loader)):
            input[0][0] = Variable(input[0][0].cuda())
            input[0][1] = Variable(input[0][1].cuda())
            #input[0][3] = Variable(input[0][3].cuda())
            #input[0][4] = Variable(input[0][4].cuda())
            input[2][1] = Variable(input[2][1].cuda())
            output = model(input)
            _, predicted = torch.max(output[0], 1)
            labels = output[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if bidx==0:
                predict = predicted
                label = labels
            else:
                predict = torch.cat((predict,predicted),0)
                label = torch.cat((label,labels),0)
        acc = 100.0 * correct / total
        print('Accuracy of the network on the test images: %f %%' % (acc))
        f1 = 100.0 * f1_score(label.cpu().numpy(), predict.cpu().numpy(), average='macro')
        pre = 100.0 * precision_score(label.cpu().numpy(), predict.cpu().numpy(), average='macro')
        rec = 100.0 * recall_score(label.cpu().numpy(), predict.cpu().numpy(), average='macro')
        acc = 100.0 * accuracy_score(label.cpu().numpy(), predict.cpu().numpy())
        print('Accuracy of the network on the test images: %f %%' % (acc))
        print('Precision of the network on the test images: %f %%'% (pre))
        print('F1 of the network on the test images: %f %%' % (f1))
        print('Recall of the network on the test images: %f %%' % (rec))
        if (bidx % self.args.print_freq) == 0 and self.summary_writer is not None:
            self.summary_writer.add_scalar('Acc', acc, epoch)
        cur_score = acc
        if(cur_score>self.best_score):
            self.best_score = cur_score
            with open(os.path.join(self.repo_path,'args_{}'.format(self.args.method)+'.json'),'w') as f:
                json.dump(vars(self.args),f,indent=4,ensure_ascii=False)
            state = {'score':self.best_score}
            try:
                torch.save(model.state_dict(),os.path.join(self.repo_path,'best_model_{}_{}_new'.format(self.args.method,str(acc))+'.pth'))
            except Exception as e:
                print(e)
                raise OSError("Something is wrong!!")
