import os
import json
import random
import argparse
from collections import Counter
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from math import ceil
from time import time
from sklearn.metrics import accuracy_score
from data import PacketDataset as Dataset
from utils import p_log, deal_results, set_log_file
from model import EBSNN_GRU, EBSNN_LSTM, FocalLoss
from torch.utils.tensorboard import SummaryWriter

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(model, args, log_writer):
    
    train_dataset = Dataset(args, 'train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                    sampler=train_sampler, num_workers=2, 
                                    drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(
        args.num_classes, args.device, alpha=train_dataset.alpha,
        gamma=args.gamma, size_average=True)
    criterion.to(args.device)

    best_results = {'acc': 0., 'epoch': 0, 'results': None}  # eval

    for epoch in range(args.epochs):

        train_loss = 0.0
        train_acc = 0.0
        y = []
        y_hat = []

        logger.info("epoch {}, total steps {}".format(epoch, len(train_dataloader)))

        for idx, batch in enumerate(train_dataloader):

            batch_X, batch_y = batch
            y += batch_y.tolist()
            batch_X, batch_y = batch_X.to(args.device), batch_y.to(args.device)
            batch_X = batch_X.view(args.batch_size, -1, args.segment_len)    # requirement of the model
            
            model.train()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if args.calculate_train_acc:
                model.eval()
                with torch.no_grad():
                    out = model(batch_X)
                    y_hat += out.max(1)[1].tolist()
                    train_acc += accuracy_score(batch_y.tolist(), out.max(1)[1].tolist())

            train_loss += loss.item()

            if (idx + 1) % args.logging_steps == 0:
                # FIXME: inaccurate training loss
                logger.info("step {}, train loss - {}, train acc - {}".format(
                                        idx + 1, 
                                        round(train_loss / (idx + 1), 4),
                                        round(train_acc / (idx + 1), 4)))
        
        train_loss = train_loss / len(train_dataloader)
        train_acc = train_acc / len(train_dataloader)

        eval_loss, eval_acc, eval_results = evaluate(model, args, test=False)

        logger.info("*" * 20 + '\n\teval_loss - {}, eval_acc - {}\n'.format(
                                        round(eval_loss, 4),
                                        round(eval_acc, 4)
                                    ))

        log_writer.add_scalar('train_loss', train_loss, epoch)
        log_writer.add_scalar('train_acc', train_acc, epoch)
        log_writer.add_scalar('eval_acc', eval_acc, epoch)
        log_writer.add_scalar('eval_loss', eval_loss, epoch)
        class_reports = deal_results(*eval_results)

        try:
            for lbl in class_reports:
                if isinstance(class_reports[lbl], dict):
                    log_writer.add_scalars(
                        'eval_metrics_label{}'.format(lbl),
                        class_reports[lbl], epoch)
                elif isinstance(class_reports[lbl], float):
                    # duplicated with eval_acc
                    pass
        except Exception as e:
            p_log('Exception: {}'.format(e))
            traceback.print_exc()
            p_log('ignore exception, please fixme')
        
        if eval_acc > best_results['acc']:
            best_results['acc'] = eval_acc
            best_results['epoch'] = epoch
            best_results['results'] = class_reports
            model_best = model.state_dict(), eval_acc, epoch
        else:
            pass
    
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'checkpoint-last.pt'))
    torch.save(model_best[0], 
        os.path.join(args.output_dir, 'checkpoint-best-epoch_{}-acc_{:.4f}.pt'.format(
                                                model_best[2], model_best[1])))
    
    log_writer.add_graph(model, input_to_model=batch_X)
    # log_writer.export_scalars_to_json(os.path.join(args.output_dir, "all_scalars.json"))

    # record hyperparams
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f)
    

def evaluate(model, args, test=False):

    # evaluate when training
    eval_dataset = Dataset(args, 'val')
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, 
                            sampler=eval_sampler, num_workers=2,
                            drop_last=True)


    criterion = FocalLoss(args.num_classes, args.device, 
                            eval_dataset.alpha, args.gamma, True)
    criterion.to(args.device)

    total_loss = 0
    y_hat, y = [], []

    time_logs = []

    for idx, batch in enumerate(eval_dataloader):
        batch_X, batch_y = batch

        batch_X = batch_X.to(args.device)
        batch_y = batch_y.to(args.device)
        batch_X = batch_X.view(args.batch_size, -1, args.segment_len)    # requirement of the model

        s_t = time()
        model.eval()
        with torch.no_grad():
            out1 = model(batch_X)
            time_logs.append(time() - s_t)
            # p_log('DEBUG: out1 in batch {}: {} (shape: {})'.format(out1, i, out1.shape))
            # the loss for flow classification when test is different to the one when train
            loss = criterion(out1, batch_y)
            total_loss += loss.item()

            # ----------------
            # torch favor of argmax...
            if not test or not args.flow:
                y_hat += out1.max(1)[1].tolist()
                y += batch_y.tolist()
            else:
                assert len(set(batch_y.tolist())) == 1, 'one batch stands for '
                'one flow when test! unexpected batch_y: {}'.format(batch_y)
                if args.aggregate == 'count_max':
                    # aggregate strategy: count_max
                    cnt = Counter(out1.max(1)[1].tolist())
                    cnt = [[v, k] for k, v in cnt]
                    cnt.sort(key=lambda x: x[0], reverse=True)
                    y_hat += [int(cnt[0][1]), ]
                else:
                    # another aggregate strategy: sum_max
                    y_hat += [int(torch.sum(out1, 0).max(0)[1].tolist()), ]
                y += [int(batch_y[0].tolist()), ]

    total_loss = total_loss / len(eval_dataloader)
    y = np.array(y)
    y_hat = np.array(y_hat)
    # p_log('DEBUG: y_hat: {} (shape: {})'.format(y_hat, y_hat.shape))
    # p_log('DEBUG: y: {} (shape: {})'.format(y, y.shape))
    accuracy = accuracy_score(y, y_hat)
    # p_log('DEBUG inference time per batch:',
    #       str(sum(time_logs) / len(time_logs)))

    return total_loss, accuracy, [y, y_hat]


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use [default: GPU 0]')
    
    # model arguments
    parser.add_argument(
        '--model', default='EBSNN_LSTM',
        help='Model name: EBSNN_LSTM or EBSNN_GRU [default: EBSNN_LSTM]')
    parser.add_argument('--embedding_dim', type=int, default=257,
                        help='embedding dimenstion [default 257]')
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--rnn_dim", default=100, type=int)
    
    # training arguments
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size [default: 32]')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Epochs [default: 50]')
    parser.add_argument(
        '--flow', action='store_true',
        help='if flow classification?')
    parser.add_argument(
        '--gamma', type=float, default=2,
        help='gamma for focal loss [default 2]')
    parser.add_argument('--test_percent', type=float, default=0.2,
                        help='test percent [default 0.2]')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate [default 0.001]')
    # parser.add_argument('--patience', type=int, default=100,
    #                     help='patience epochs for early_stopping [default 100]')
    parser.add_argument("--max_length", default=1500, type=int)

    parser.add_argument("--dataset", default='d1', type=str, choices=['d1', 'd2'])
    parser.add_argument(
        '--labels', type=str,
        default='skype,pplive,baidu,tudou,weibo,thunder,youku,itunes,'
        'taobao,qq,gmail,sohu',
        help='names of labels, seperated by ",", modify it if you need')
    # top_k, aggregate stragety
    parser.add_argument(
        '--first_k_packets', type=int, default=3,
        help='first_k_packets for flow classification, value must in '
        'range [1, threshold] [default 3]')
    parser.add_argument(
        '--aggregate', type=str, default='sum_max', choices=['sum_max', 'count_max'],
        help='aggregate strategy for flow classification, sum_max or '
        'count_max [default sum_max]')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--shuffle', action='store_true', help='if shuffle dataset')
    parser.add_argument('--no_bidirectional', action='store_true',
                        help='if bi-RNN')
    parser.add_argument('--segment_len', type=int, default=8,
                        help='the length of segment')
    parser.add_argument('--test_cycle', type=int, default=1,
                        help='test cycle')
    

    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--logging_steps", default=20, type=int)
    parser.add_argument(
        '--log_filename', type=str,
        default='log_20/log_train.txt',
        help='file name of log'
    )
    parser.add_argument("--seed", default=42, type=int)
    

    # extra arguments
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--calculate_train_acc", action='store_true', 
                        help='need to calculate acc in eval mode (slow)')
    parser.add_argument("--do_eval", action='store_true')


    args = parser.parse_args()

    return args


def main():

    args = get_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    LABELS = {v: k for k, v in enumerate(args.labels.split(','))}   # FIXME: modify this
    # args.num_classes = len(LABELS)
    args.num_classes = 29   # NOTE: RuntimeError in CUDA is caused by inconsistent classes
    args.alpha = None   # debug
    args.n_gpu = 1

    set_seed(args)

    set_log_file(args.log_filename)

    MODEL_CLASS = {
        'EBSNN_LSTM': EBSNN_LSTM, 'EBSNN_GRU': EBSNN_GRU
    }[args.model]
    log_writer = SummaryWriter()
    model = MODEL_CLASS(args.num_classes, args.embedding_dim, args.device,
                  bidirectional=not args.no_bidirectional,
                  segment_len=args.segment_len,
                  dropout_rate=args.dropout,
                  rnn_dim=args.rnn_dim)
    model.to(args.device)
    
    if args.do_train:
        train(model, args, log_writer)
    
    if args.do_eval:
        evaluate(model, args)
    
    log_writer.close()


if __name__ == '__main__':
    main()