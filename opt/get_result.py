import os
import sys
sys.path.append('../')
import pickle
import random
import argparse
from tqdm import tqdm
import torch
from datetime import datetime
from pprint import pprint
import time
import easydict
import json
from Model.TIRG import TIRG
from Model.MultiTrans import MultitransModel
from Model.ViLBERT import ViLBERT
from Model.text_only import TextOnlyModel
from Model.image_only import ImageOnlyModel
from Model.attribute_only import AttributeOnlyModel
from Model.Sum import SumModel
from Model.Concat import Concat
from Model.composeae import ComposeAE
from Model.cross_attention import Combine
from Model.single_attention import TToA, AToT
from Model.imgattr import ImgAttr
from Model.M3HATT import M3HATT
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import recall_score,f1_score,precision_score
def init_env():
    state = {k:v for k,v in args._get_kwargs()}
    pprint(state)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manualSeed)
        torch.backends.cudnn.benchmark = True

score = dict()
hyperopt = dict()

def main(args):
    init_env()
    date_key = str(datetime.now().strftime('%Y%m%d%H%M%S'))[2:]
    print 'Load model: {}'.format(args.method)
    root_path = '../repo/{}'.format(args.method)
    with open(os.path.join(root_path,'args_{}'.format(args.method)+'.json'),'r') as f:
        largs = json.load(f)
        largs = easydict.EasyDict(largs)
        pprint(largs)
    from Preprocess.dataset import MyTrainTestDataset
    train_dataset = MyTrainTestDataset(
        data_root=args.data_root,
        image_root=args.image_root,
        image_size=largs.image_size,
        split='train')
    val_dataset = MyTrainTestDataset(
        data_root=args.data_root,
        image_root=args.image_root,
        image_size=largs.image_size,
        split='val'
    )
    val_loader = val_dataset.get_loader(batch_size=largs.batch_size)
    test_dataset = MyTrainTestDataset(
        data_root=args.data_root,
        image_root=args.image_root,
        image_size=largs.image_size,
        split='test'
    )
    test_loader = test_dataset.get_loader(batch_size=largs.batch_size)
    if args.method == 'text-only':
        model = TextOnlyModel(args=largs,backbone=largs.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=largs.text_method,fdims=largs.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=largs.stack_num)
    elif args.method == 'tirg':
        model = TIRG(args=largs,backbone=largs.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=largs.text_method,fdims=largs.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=largs.stack_num)
    elif args.method == 'image-only':
        model = ImageOnlyModel(args=largs,backbone=largs.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=largs.text_method,fdims=largs.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=largs.stack_num)
    elif args.method == 'attribute-only':
        model = AttributeOnlyModel(args=largs,backbone=largs.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=largs.text_method,fdims=largs.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=largs.stack_num)
    elif args.method == 'concat':
        model = Concat(args=largs,backbone=largs.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=largs.text_method,fdims=largs.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=largs.stack_num)
    elif args.method == 'sum':
        model = SumModel(args=largs,backbone=largs.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=largs.text_method,fdims=largs.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=largs.stack_num)
    elif args.method == 'composeae':
        model = ComposeAE(args=largs,backbone=largs.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=largs.text_method,fdims=largs.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=largs.stack_num)
    elif args.method == 'attribute':
        model = Combine(args=largs,backbone=largs.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=largs.text_method,fdims=largs.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=largs.stack_num)
    elif args.method == 't2a':
        model = TToA(args=largs,backbone=largs.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=largs.text_method,fdims=largs.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=largs.stack_num)
    elif args.method == 'a2t':
        model = AToT(args=largs,backbone=largs.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=largs.text_method,fdims=largs.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=largs.stack_num)
    elif args.method == 'imgattr':
        model = ImgAttr(args=largs,backbone=largs.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=largs.text_method,fdims=largs.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=largs.stack_num)
    elif args.method == 'multitrans':
        model = MultitransModel(args=largs,backbone=largs.backbone,texts=train_dataset.get_all_texts(),
                              text_method=largs.text_method,fdims=largs.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=largs.stack_num,attrs=train_dataset.get_all_attribute_texts())
    elif args.method == 'vilbert':
        model = ViLBERT(args=largs,backbone=largs.backbone,texts=train_dataset.get_all_texts(),
                              text_method=largs.text_method,fdims=largs.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=largs.stack_num,attrs=train_dataset.get_all_attribute_texts())
    elif args.method == 'm3hatt':
        model = M3HATT(args=largs,backbone=largs.backbone,texts=train_dataset.get_all_texts(),
                              text_method=largs.text_method,fdims=largs.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=largs.stack_num,attrs=train_dataset.get_all_attribute_texts())
    else:
        raise NotImplementedError()
    model.load_state_dict(torch.load(os.path.join(root_path,'best_model_{}'.format(largs.method)+'.pth')))
    model = model.cuda()
    model.eval()
    scores = []
    query_ids = []
    total = 0.
    correct = 0.
    recall = 0.
    precision = 0.
    f1 = 0.
    for bidx, input in enumerate(tqdm(test_loader,desc='Test')):
        with torch.no_grad():
            input[0][0] = Variable(input[0][0].cuda())
            input[0][3] = Variable(input[0][3].cuda())
            input[0][4] = Variable(input[0][4].cuda())
            input[0][1] = Variable(input[0][1].cuda())
            input[2][1] = Variable(input[2][1].cuda())
            output = model(input)
        _, predicted = torch.max(output[0], 1)
        predicted1 = torch.nn.functional.softmax(output[0], dim=1)
        labels = output[1]
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        #_precision = precision_score(labels.cpu().numpy(), predicted.cpu().numpy(),average='macro')
        #_recall = recall_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
       # _f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
        #recall+=_recall
        #precision+=_precision
        #f1+=_f1
        _predicted = predicted1.cpu().numpy()
        _label = labels.cpu().numpy()
        if bidx == 0:
            label = _label
            pre = _predicted
        else:
            label = np.concatenate((label,_label))
            pre = np.concatenate((pre,_predicted))
        # for i in range(predicted.size(0)):
        #     _score = predicted[i].squeeze().cpu().numpy()
        #     _qid = input[0][2][i]
        #     scores.append(_score)
        #     query_ids.append(_qid)
    acc = 100.0 * correct / total
    #recall = recall/bidx
   # precision = precision/bidx
   # f1 = f1/bidx
    precision = precision_score(label,pre.argmax(axis=1),average='macro')
    recall = recall_score(label,pre.argmax(axis=1),average='macro')
    f1 = f1_score(label,pre.argmax(axis=1),average='macro')
    print('Accuracy of the network on the test images: %f %%' % (acc))
    print('Precision of the network on the test images: %f %%' % (precision))
    print('F1 score of the network on the test images: %f %%' % (f1))
    print('Recall of the network on the test images:  %f %%' % (recall))
    # *****write results*******
    scores = np.asarray(scores)
    with open('results/{}_label'.format(largs.method),'wb') as f:
        pickle.dump(label,f)
    with open('results/{}_predicted'.format(largs.method),'wb') as f:
        pickle.dump(pre,f)

# ------------------------------- val result -----------------------------------
    scores = []
    total = 0.
    correct = 0.
    recall = 0.
    precision = 0.
    f1 = 0.
    for bidx, input in enumerate(tqdm(val_loader, desc='Val')):
        with torch.no_grad():
            input[0][0] = Variable(input[0][0].cuda())
            input[0][1] = Variable(input[0][1].cuda())
            input[0][3] = Variable(input[0][3].cuda())
            input[0][4] = Variable(input[0][4].cuda())
            input[2][1] = Variable(input[2][1].cuda())
            output = model(input)
        _, predicted = torch.max(output[0], 1)
        predicted1 = torch.nn.functional.softmax(output[0], dim=1)
        labels = output[1]
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # predicted = torch.nn.functional.softmax(output[0],dim=1)
       # _precision = precision_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
        #_recall = recall_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
        #_f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro')
        #recall += _recall
        #precision += _precision
        #f1 += _f1
        _predicted = predicted1.cpu().numpy()
        _label = labels.cpu().numpy()
        if bidx == 0:
            label = _label
            pre = _predicted
        else:
            label = np.concatenate((label,_label))
            pre = np.concatenate((pre,_predicted))
    
    precision = precision_score(label,pre.argmax(axis=1),average='macro')
    recall = recall_score(label,pre.argmax(axis=1),average='macro')
    f1 = f1_score(label,pre.argmax(axis=1),average='macro')
    acc = 100.0 * correct / total
    #recall = recall / bidx
    #precision = precision / bidx
    #f1 = f1 / bidx
    print('Accuracy of the network on the validation images: %f %%' % (acc))
    print('Precision of the network on the validation images: %f %%' % (precision))
    print('F1 score of the network on the validation images: %f %%' % (f1))
    print('Recall of the network on the validation images:  %f %%' % (recall))
    scores = np.asarray(scores)
    with open('results/{}_label_val'.format(largs.method),'wb') as f:
        pickle.dump(label,f)
    with open('results/{}_predicted_val'.format(largs.method),'wb') as f:
        pickle.dump(pre,f)
    # hyperopt = {
    #     'score':scores,
    #     'query_ids':query_ids
    # }


# ------------------------------- dump final result -----------------------------------


    # print 'Dump the result matrix to .pkl file'
    # final_path = './output_score/{}_{}_{}'.format(date_key,args.expr_name,args.method)
    # if not os.path.exists(final_path):
    #     os.makedirs(final_path)
    # with open(os.path.join(final_path,'hyperopt.pkl'),'wb') as f:
    #     pickle.dump(hyperopt, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('test')
    parser.add_argument('--gpu_id', default='2',type=str)
    parser.add_argument('--manualSeed',type=int,default=int(time.time()))
    parser.add_argument('--data_root',type=str,default='../../data/Mydataset/Com/')
    parser.add_argument('--image_root',type=str,default='../../data/Mydataset/Com/images')
    parser.add_argument('--image_size',default=224)
    parser.add_argument('--method',type=str,default='tirg')
    args, _ = parser.parse_known_args()

    main(args)
