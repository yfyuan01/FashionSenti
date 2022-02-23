import os
import sys
import time
import random
import argparse
from pprint import pprint
import torch
from Preprocess.dataset import TwitterTrainValDataset, TwitterTestDataset,Bt4saTrainTestDataset,MyTrainTestDataset,TWMMS1TrainTestDataset
from Model.text_only import TextOnlyModel
from Model.image_only import ImageOnlyModel
from Model.attribute_only import AttributeOnlyModel
from Model.TIRG import TIRG
from Model.M3HATT import M3HATT
from Model.Sum import SumModel
#from Model.composeae import ComposeAE
from Model.Concat import Concat
from Model.ViLBERT import ViLBERT
from Model.cross_attention import Combine
from tensorboardX import SummaryWriter
from Preprocess.runner import Trainer,Evaluator
from Model.MultiTrans import MultitransModel
from Model.imgattr import ImgAttr
from Model.single_attention import TToA,AToT
def init_env():
    args = parser.parse_args()
    state = {k:v for k,v in args._get_kwargs()}
    pprint(state)
    args.lr_decay_steps = [int(x) for x in args.lr_decay_steps.strip().split(',')]

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    if args.manualSeed is None:
        args.manualSeed = random.randint(1,10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manualSeed)
        #torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def main():
    init_env()
    #args = parser.parse_args()
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
    elif args.dataset=='mydataset':
        train_dataset = MyTrainTestDataset(
            data_root=args.data_root,
            image_root=args.image_root,
            image_size=args.image_size,
            split='train',
        )
        train_loader = train_dataset.get_loader(batch_size=args.batch_size)
        test_dataset = MyTrainTestDataset(
            data_root=args.data_root,
            image_size=args.image_size,
            image_root=args.image_root,
            split='test',
        )
        test_loader = test_dataset.get_loader(batch_size=args.batch_size)
    elif args.dataset=='twmss1':
        train_dataset = TWMMS1TrainTestDataset(
            data_root=args.data_root,
            image_root=args.image_root,
            image_size=args.image_size,
            split='train',
        )
        train_loader = train_dataset.get_loader(batch_size=args.batch_size)
        test_dataset = TWMMS1TrainTestDataset(
            data_root=args.data_root,
            image_size=args.image_size,
            image_root=args.image_root,
            split='test',
        )
        test_loader = test_dataset.get_loader(batch_size=args.batch_size)
    else:
        train_dataset = TwitterTrainValDataset(
            data_root = args.data_root,
            image_root=args.image_root,
            tag_root=args.tag_root,
            image_size = args.image_size,
            split = 'train',
        )
        train_loader = train_dataset.get_loader(batch_size=args.batch_size,)
        test_dataset = TwitterTestDataset(
                data_root = args.data_root,
                image_root=args.image_root,
                tag_root=args.tag_root,
                image_size = args.image_size,
                split = 'val',
        )
        test_loader = test_dataset.get_loader(batch_size=args.batch_size)

    if args.method == 'text-only':
        model = TextOnlyModel(args=args,backbone=args.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=args.text_method,fdims=args.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=args.stack_num)
    elif args.method == 'attribute-only':
        model = AttributeOnlyModel(args=args,backbone=args.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=args.text_method,fdims=args.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=args.stack_num)
    elif args.method == 'tirg':
        model = TIRG(args=args,backbone=args.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=args.text_method,fdims=args.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=args.stack_num)
    elif args.method == 'vilbert':
        model = ViLBERT(args=args,backbone=args.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=args.text_method,fdims=args.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=args.stack_num)
    elif args.method == 'composeae':
        model = ComposeAE(args=args,backbone=args.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=args.text_method,fdims=args.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=args.stack_num)
    elif args.method == 'imgattr':
        model = ImgAttr(args=args,backbone=args.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=args.text_method,fdims=args.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=args.stack_num)
    elif args.method == 'image-only':
        model = ImageOnlyModel(args=args,backbone=args.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=args.text_method,fdims=args.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=args.stack_num)
    elif args.method == 'concat':
        model = Concat(args=args,backbone=args.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=args.text_method,fdims=args.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=args.stack_num)
    elif args.method == 'sum':
        model = SumModel(args=args,backbone=args.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=args.text_method,fdims=args.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=args.stack_num)
    elif args.method == 'attribute':
        model = Combine(args=args,backbone=args.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=args.text_method,fdims=args.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=args.stack_num)
    elif args.method == 't2a':
        model = TToA(args=args,backbone=args.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=args.text_method,fdims=args.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=args.stack_num)
    elif args.method == 'a2t':
        model = AToT(args=args,backbone=args.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=args.text_method,fdims=args.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=args.stack_num)
    elif args.method == 'm3hatt':
        model = M3HATT(args=args,backbone=args.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=args.text_method,fdims=args.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=args.stack_num)
    elif args.method == 'multitrans':
        model = MultitransModel(args=args,backbone=args.backbone,texts=train_dataset.get_all_texts(),attrs=train_dataset.get_all_attribute_texts(),
                              text_method=args.text_method,fdims=args.fdims,fc_arch='A',
                              init_with_glove=False,stack_num=args.stack_num)
    else:
        raise NotImplementedError()
    model = model.cuda()
    log_path = os.path.join('logs',args.expr_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    summary_writer = SummaryWriter(log_path)

    trainer = Trainer(args=args,data_loader=train_loader,
                      model=model,summary_writer=summary_writer)
    evaluator = Evaluator(args=args, data_loader=test_loader,
                      model=model, summary_writer=summary_writer, eval_freq=1)
    print "start training"
    for epoch in range(args.epochs):
        epoch += 1
        trainer.train(epoch)
        evaluator.test(epoch)
    print "Congrats! You just finished training."


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train')
    parser.add_argument('--gpu_id',default='0',type=str,help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--manualSeed',type=int,default=int(time.time()),help='manual seed')
    parser.add_argument('--stack_num',type=int,default=2,help='self attention block number')
    parser.add_argument('--warmup',action='store_true',help='warmup?')
    parser.add_argument('--expr_name',default='devel',type=str,help='experiment name')
    parser.add_argument('--data_root',required=False,type=str,default='../data/twmms1/')
    parser.add_argument('--text_method',default='lstm',choices=['lstm','swem','lstm-gru','encode','vilbert'],type=str)
    parser.add_argument('--fdims',default=2048,type=int,help='output feature dimensions')
    parser.add_argument('--method',default='multitrans',type=str,help='method')
    parser.add_argument('--epochs',default=100,type=int)
    parser.add_argument('--print_freq',default=1,type=int)
    parser.add_argument('--batch_size',default=64,type=int,help='train batchsize')
    parser.add_argument('--image_size',default=224,type=int,help='image size (default:16)')
    parser.add_argument('--backbone',default='resnet152',type=str)
    parser.add_argument('--normalize_scale',default=5.0,type=float)
    parser.add_argument('--lr',default=0.0011148,type=float,help='initial learning rate')
    parser.add_argument('--lrp',default=0.48,type=float,help='lrp')
    parser.add_argument('--lr_decay_factor',default=0.4747,type=float)
    parser.add_argument('--lr_decay_steps',default='10,20,30,40,50,60,70',type=str)
    parser.add_argument('--image_root',type=str,default='../data/twmms1/images')
    parser.add_argument('--dataset',default='mydataset',choices=['bt4sa','twitter','mydataset','twmss1'])
    parser.add_argument('--tag_root', default='../data/Twitter_Dataset/twitter/groundTruthLabel.txt', type=str)
    args, _ = parser.parse_known_args()
    main()
