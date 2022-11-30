import argparse
import logging
import traceback
import os
import time
import math
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import accuracy_score

from model import EBSNN_GRU, EBSNN_LSTM, FocalLoss, KDLoss
from data import PacketDataset, FlowDataset
from utils import RunningAverage, deal_results

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


# Defining train_kd & train_and_evaluate_kd functions
def train_kd(model, teacher_model, args):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        dataloader: 
        metrics: (dict) 
        params: (Params) hyperparameters
    """

    dataset = PacketDataset(args, 'train')
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                            sampler=sampler, num_workers=2, 
                            drop_last=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = KDLoss(args.kd_alpha, args.kd_temperature, args).to(args.device)
    criterion.to(args.device)

    # set model to training mode
    model.train()
    teacher_model.eval()

    best_results = {'acc': 0., 'epoch': 0, 'results': None}  # eval
    
    for epoch in range(args.epochs):

        loss_avg = RunningAverage()
        logger.info("epoch {}, total steps {}".format(epoch, len(dataloader)))

        # Use tqdm for progress bar
        with tqdm(total=len(dataloader)) as t:
            for i, (train_batch, labels_batch) in enumerate(dataloader):

                train_batch, labels_batch = train_batch.to(args.device), \
                                            labels_batch.to(args.device)

                # compute model output, fetch teacher output, and compute KD loss
                output_batch = model(train_batch)

                # get one batch output from teacher_outputs list

                with torch.no_grad():
                    output_teacher_batch = teacher_model(train_batch)
                output_teacher_batch = output_teacher_batch.to(args.device)

                loss = criterion(output_batch, labels_batch, output_teacher_batch)

                # clear previous gradients, compute gradients of all variables wrt loss
                optimizer.zero_grad()
                loss.backward()

                # performs updates using calculated gradients
                optimizer.step()

                loss_avg.update(loss.data[0])

                t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
                t.update()

        eval_loss, eval_acc, eval_results = evaluate_kd(model, args, test=False)
        logger.info("*" * 20 + '\n\teval_loss - {}, eval_acc - {}\n'.format(
                                        round(eval_loss, 4),
                                        round(eval_acc, 4)
                                    ))
        
        class_reports = deal_results(*eval_results)

        # try:
        #     for lbl in class_reports:
        #         if isinstance(class_reports[lbl], dict):
        #             log_writer.add_scalars(
        #                 'eval_metrics_label{}'.format(lbl),
        #                 class_reports[lbl], epoch)
        #         elif isinstance(class_reports[lbl], float):
        #             # duplicated with eval_acc
        #             pass
        # except Exception as e:
        #     p_log('Exception: {}'.format(e))
        #     traceback.print_exc()
        #     p_log('ignore exception, please fixme')
        
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


def evaluate_kd(model, args, test=False):

    # evaluate when training
    if args.flow and test:
        eval_dataset = FlowDataset(args, 'val')
        args.batch_size = 1
    else:
        eval_dataset = PacketDataset(args, 'val')
    
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
    parser.add_argument('--data_dir', default='../data', type=str)
    parser.add_argument('--student_config', type=str)
    parser.add_argument('--teacher_config', type=str)
    parser.add_argument('--model_dir', default='experiments/base_model',
                        help="Directory containing params.json")
    parser.add_argument('--restore_file', default=None,
                        help="Optional, name of the file in --model_dir \
                        containing weights to reload before training")  # 'best' or 'train'
    parser.add_argument("--logging_steps", default=20, type=int)
    parser.add_argument(
        '--log_filename', type=str,
        default='log_20/log_train.txt',
        help='file name of log'
    )
    
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
    parser.add_argument('--epochs', type=int, default=50,
                        help='Epochs [default: 50]')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate [default 0.001]')
    parser.add_argument("--max_length", default=1500, type=int)

    parser.add_argument("--dataset", default='d1', type=str, choices=['d1', 'd2'])
    parser.add_argument(
        '--labels', type=str,
        default='skype,pplive,baidu,tudou,weibo,thunder,youku,itunes,'
        'taobao,qq,gmail,sohu',
        help='names of labels, seperated by ",", modify it if you need')
    # top_k, aggregate strategy
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
    
    # kd args
    parser.add_argument("--kd_alpha", default=0.0, type=float)
    parser.add_argument("--kd_temperature", default=1.0, type=float)
    parser.add_argument("--focal", action='store_true')

    # extra arguments
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')


    return parser.parse_args()


def get_model(config_path, model_dir=None, restore_file=None):
    "TODO: load model by means of config"
    # model_type, dim, segment_len, bidirectional,

    with open(config_path, 'r') as f:
        config = json.load(f)

    if model_dir:   # teacher model
        assert restore_file

    else:           # student model
        pass


def main():

    args = get_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.num_classes = 29   # d1

    if args.do_train:

        student_model = get_model(args.student_config)
        teacher_model = get_model(args.teacher_config, args.model_dir, args.restore_file)

        train_kd(student_model, teacher_model, args)
    
    if args.do_eval:

        model = get_model(args.teacher_config, args.model_dir, args.restore_file)
        evaluate_kd(model, args, test=True)


if __name__ == '__main__':
    main()