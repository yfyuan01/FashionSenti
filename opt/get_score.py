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
from Model.text_only import TextOnlyModel
from Model.image_only import ImageOnlyModel
from Model.Sum import SumModel
from Model.Concat import Concat
from Model.cross_attention import Combine
from torch.autograd import Variable
import numpy as np
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
    print 'Load model: {}'.format(args.expr_name)
    root_path = '../repo/{}'.format(args.expr_name)
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
    test_dataset = MyTrainTestDataset(
        data_root=args.data_root,
        image_root=args.image_root,
        image_size=largs.image_size,
        split='test'
    )
    test_loader = test_dataset.get_loader(batch_size=largs.batch_size)
    if args.method == 'text-only':
        model = TextOnlyModel(args=largs,backbone=largs.backbone,texts=train_dataset.get_all_texts(),
                              text_method=largs.text_method,fdims=largs.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=largs.stack_num)
    elif args.method == 'tirg':
        model = TIRG(args=largs,backbone=largs.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=largs.text_method,fdims=largs.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=largs.stack_num)
    elif args.method == 'image-only':
        model = ImageOnlyModel(args=largs,backbone=largs.backbone,texts=train_dataset.get_all_texts(),
                              text_method=largs.text_method,fdims=largs.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=largs.stack_num)
    elif args.method == 'concat':
        model = Concat(args=largs,backbone=largs.backbone,texts=train_dataset.get_all_texts(),
                              text_method=largs.text_method,fdims=largs.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=largs.stack_num)
    elif args.method == 'sum':
        model = SumModel(args=largs,backbone=largs.backbone,texts=train_dataset.get_all_texts(),
                              text_method=largs.text_method,fdims=largs.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=largs.stack_num)
    elif args.method == 'attribute':
        model = Combine(args=largs,backbone=largs.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=largs.text_method,fdims=largs.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=largs.stack_num)
    elif args.method == 'multitrans':
        model = MultitransModel(args=largs,backbone=largs.backbone,texts=train_dataset.get_all_texts(),
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
    for bidx, input in enumerate(tqdm(test_loader,desc='Test')):
        with torch.no_grad():
            input[0][0] = Variable(input[0][0].cuda())
            input[0][1] = Variable(input[0][1].cuda())
            input[2][1] = Variable(input[2][1].cuda())
            output = model(input)
        _, predicted = torch.max(output[0], 1)
        labels = output[1]
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted = torch.nn.functional.softmax(output[0],dim=1)

        for i in range(predicted.size(0)):
            _score = predicted[i].squeeze().cpu().numpy()
            _qid = input[0][2][i]
            scores.append(_score)
            query_ids.append(_qid)
    acc = 100.0 * correct / total
    print('Accuracy of the network on the test images: %f %%' % (acc))
    scores = np.asarray(scores)
    hyperopt = {
        'score':scores,
        'query_ids':query_ids
    }
    print 'Dump the result matrix to .pkl file'
    final_path = './output_score/{}_{}_{}'.format(date_key,args.expr_name,args.method)
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    with open(os.path.join(final_path,'hyperopt.pkl'),'wb') as f:
        pickle.dump(hyperopt, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('test')
    parser.add_argument('--gpu_id', default='0',type=str)
    parser.add_argument('--manualSeed',type=int,default=int(time.time()))
    parser.add_argument('--data_root',type=str,default='../../data/Mydataset/')
    parser.add_argument('--expr_name',default='devel',type=str)
    parser.add_argument('--image_root',type=str,default='../../data/Mydataset/images')
    parser.add_argument('--image_size',default=224)
    parser.add_argument('--method',type=str,default='multitrans')
    args, _ = parser.parse_known_args()

    main(args)
