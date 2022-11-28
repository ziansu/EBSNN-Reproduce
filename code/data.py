import os
import pickle
import random
from time import time
from collections import Counter
import numpy as np
import torch


def calculate_alpha(counter, mode='normal'):
    if mode == 'normal':
        alpha = torch.tensor(counter, dtype=torch.float32)
        alpha = alpha / alpha.sum(0).expand_as(alpha)
    elif mode == 'invert':
        alpha = torch.tensor(counter, dtype=torch.float32)
        alpha_sum = alpha.sum(0)
        alpha_sum_expand = alpha_sum.expand_as(alpha)
        alpha = (alpha_sum - alpha) / alpha_sum_expand
    # fill all zeros to ones
    alpha[alpha==0.] = 1.
    return alpha


class Dataset(torch.utils.data.Dataset):
    "TODO: calculate alpha"

    def __init__(self, args, file_type):

        self.file_type = file_type
        self.max_length = args.max_length
        self.max_length = int(args.max_length / args.segment_len) * args.segment_len  # easy viewing

        data_path = os.path.join(args.data_dir, f'{args.dataset}_{file_type}_dump.pkl')
        assert os.path.exists(data_path)
        with open(data_path, 'rb') as f:
            self.features = pickle.load(f)
            self.labels = pickle.load(f)
            self.label2id = pickle.load(f)
            self.id2label = pickle.load(f)
        
        self.counter = Counter(self.labels)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        x = list(self.features[index])
        x = x[:self.max_length]         # truncating
        if len(x) < self.max_length:    # padding
            x = x + [256] * (self.max_length - len(x))
        y = self.labels[index]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y)


if __name__ == '__main__':

    class PseudoArgs:
        data_dir = '../data'
        dataset = 'd1'
        max_length = 1500
        segment_len = 8

    args = PseudoArgs()

    dataset = Dataset(args, 'train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

    for X, y in dataloader:
        print(X.shape)
        print(y.shape)
        break