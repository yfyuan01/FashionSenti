import os
import sys
sys.path.append('../')
import pickle
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse
from hyperopt import hp, tpe, fmin
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
def objective_fn(W, args):
    W = np.array(W)/sum(W)
    final_matrix = {}
    labels = args.label
    for i, repo in enumerate(args.repos):
        try:
            pkl_path = os.path.join('results','{}_predicted_val'.format(repo))
            assert os.path.exists(pkl_path)
            saved_scores = pickle.load(open(pkl_path,'rb'))
        except Exception as err:
            raise OSError('{}:{}'.format(pkl_path,err))
        tag = 'score'
        if tag in final_matrix:
            final_matrix[tag] += saved_scores * W[i]
        else:
            final_matrix[tag] = saved_scores * W[i]

    y_score = final_matrix[tag]
    predicted = np.argmax(y_score, 1)
    # total = labels.shape[0]
    # print predicted.size
    # print labels.size
    acc = accuracy_score(labels,predicted)
    pre = precision_score(labels,predicted,average='macro')
    rec = recall_score(labels,predicted,average='macro')
    f1 = f1_score(labels,predicted,average='macro')
    # correct = (predicted == labels).sum().item()
    # acc = 100.0 * correct / total
    total = (acc+pre+rec+f1)/4
    # print "[{}] acc:{}".format(W,acc)
    return -total


def main(args):
    args.repos = args.repos.strip().split(',')
    space = [hp.uniform('w{}'.format(i), 0, 1) for i in range(len(args.repos))]
    print 'loading labels'
    from Preprocess.dataset import MyTrainTestDataset
    # val_dataset = MyTrainTestDataset(
    #     data_root=args.data_root,
    #     image_size=args.image_size,
    #     image_root=args.image_root,
    #     split='val',
    # )
    # val_loader = val_dataset.get_loader(batch_size=32)
    # test_dataset = MyTrainTestDataset(
    #     data_root=args.data_root,
    #     image_size=args.image_size,
    #     image_root=args.image_root,
    #     split='test',
    # )
    # test_loader = test_dataset.get_loader(batch_size=32)
    # label = []
    # for bidx, input in enumerate(tqdm(val_loader)):
    #     labels = input[2][1]
    #     for i in range(labels.size(0)):
    #         _label = labels[i].squeeze().numpy()
    #         label.append(_label)
    # label = np.asarray(label)
    with open(os.path.join('results','{}_label_val'.format(args.repos[0])),'rb') as f:
        label = pickle.load(f)
    args.label = label
    best = fmin(fn=lambda W: objective_fn(W, args),
                space=space,
                algo=tpe.suggest,
                max_evals=args.max_eval)
    print 'best: {}'.format(best)
    for i, repo in enumerate(args.repos):
        pkl_path = os.path.join('results','{}_predicted'.format(repo))
        saved_scores = pickle.load(open(pkl_path,'rb'))
        partial_scores = saved_scores*best['w'+str(i)]
        pkl_path_val = os.path.join('results', '{}_predicted_val'.format(repo))
        saved_scores_val = pickle.load(open(pkl_path_val, 'rb'))
        partial_scores_val = saved_scores_val * best['w'+str(i)]
        if i==0:
            final_score_val = partial_scores_val
            final_score = partial_scores
        else:
            final_score_val += partial_scores_val
            final_score += partial_scores
    predicted_val = final_score_val.argmax(axis=1)
    predicted = final_score.argmax(axis=1)
    recall = recall_score(label, predicted_val, average='macro')
    precision = precision_score(label, predicted_val, average='macro')
    f1 = f1_score(label, predicted_val, average='macro')
    acc = accuracy_score(label, predicted_val)
    print 'Val: acc: {}, precision: {}, recall: {}, f1: {}'.format(acc, precision, recall, f1)
    label_path = os.path.join('results','{}_label'.format(args.repos[0]))
    label1 = pickle.load(open(label_path,'rb'))
    recall = recall_score(label1,predicted,average='macro')
    precision = precision_score(label1,predicted,average='macro')
    f1 = f1_score(label1,predicted,average='macro')
    acc = accuracy_score(label1,predicted)
    print 'Test: acc: {}, precision: {}, recall: {}, f1: {}'.format(acc,precision,recall,f1)
    print 'Saving the final score.....'
    with open('results/final_predicted','wb') as f:
        pickle.dump(final_score,f)
    # date_key = str(datetime.now().strftime('%Y%m%d%H%M'))[2:]
    # splits = ['val','test']
    # for split in splits:
    #     print('Save final results for {}...'.format(split))
    #     os.system('mkdir -p output_optimize')
    #     final_score = dict()
    #     for idx, repo in enumerate(args.repos):
    #         try:
    #             pkl_path = os.path.join('output_score',repo,'hyperopt.{}.pkl'.format(split))
    #             assert os.path.exists(pkl_path)
    #             saved_scores = pickle.load(open(pkl_path,'rb'))
    #         except Exception as err:
    #             raise OSError('{},{}'.format(pkl_path,err))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Bayes Optimization on the scores')
    parser.add_argument('--data_root',default='../../data/Mydataset/Com',type=str)
    parser.add_argument('--repos',default='devel',type=str)
    parser.add_argument('--max_eval',default=400,type=int)
    parser.add_argument('--image_root',type=str,default='../../data/Mydataset/Com/images')
    parser.add_argument('--image_size',default=224)
    args,_ = parser.parse_known_args()
    main(args)
