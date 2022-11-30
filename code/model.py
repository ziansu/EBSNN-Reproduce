import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import p_log


class EBSNN_GRU(nn.Module):
    def __init__(self, num_class, embedding_dim, device,
                 segment_len=8, bidirectional=True,
                 dropout_rate=0.5):
        super(EBSNN_GRU, self).__init__()
        self.num_class = num_class
        self.embedding_dim = embedding_dim
        self.device = device
        self.segment_len = segment_len
        self.rnn_dim = 100
        # if bi-direction
        self.rnn_directions = 2 if bidirectional else 1

        # 256 is 'gg', will be set [0,0..0]
        self.byte_embed = nn.Embedding(257, self.embedding_dim, padding_idx=256)
        # to one-hot
        self.byte_embed.requires_grad = True
        self.rnn1 = nn.GRU(input_size=self.embedding_dim,
                           hidden_size=self.rnn_dim,
                           batch_first=True, dropout=dropout_rate,
                           bidirectional=(self.rnn_directions == 2))
        self.rnn2 = nn.GRU(self.rnn_directions * self.rnn_dim,
                           self.rnn_dim, batch_first=True,
                           dropout=dropout_rate,
                           bidirectional=(self.rnn_directions == 2))
        # consider other initialization
        self.hc1 = nn.parameter.Parameter(torch.randn(self.rnn_dim, 1))
        self.hc2 = nn.parameter.Parameter(torch.randn(self.rnn_dim, 1))
        
        self.fc1 = nn.Linear(self.rnn_dim * self.rnn_directions,
                             self.rnn_dim)
        self.fc2 = nn.Linear(self.rnn_dim * self.rnn_directions,
                             self.rnn_dim)
        self.fc3 = nn.Linear(self.rnn_dim * self.rnn_directions,
                             self.num_class)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        x: b * l * 8
        lengths: (b,), every batch's length
        """
        batch_size = x.size(0)
        x = self.byte_embed(x)  # b * l * 8 * 257

        out1, (h_n, c_n) = self.rnn1(x.view(-1, self.segment_len, self.embedding_dim))
        # bl * 8 * 100
        h = torch.tanh(self.fc1(out1))
        weights = (torch.matmul(h, self.hc1)).view(-1, self.segment_len)
        weights = F.softmax(weights, dim=1)
        weights = weights.view(-1, 1, self.segment_len)
        # b * l * 200
        out2 = torch.matmul(weights, out1).view(
            batch_size, -1, self.rnn_dim * self.rnn_directions)

        out3, (h1_n, h2_n) = self.rnn2(out2)  # out3: b * l * 200
        h2 = torch.tanh(self.fc2(out3))
        weights2 = F.softmax((torch.matmul(h2, self.hc2)).view(
            batch_size, -1), dim=1).view(batch_size, 1, -1)
        out4 = torch.matmul(weights2, out3).view(
            batch_size, self.rnn_dim * self.rnn_directions)

        out = self.dropout(self.fc3(out4))
        return out


class EBSNN_LSTM(nn.Module):
    def __init__(self, num_class, embedding_dim, device,
                 bidirectional=True, segment_len=8, dropout_rate=0.5):
        super(EBSNN_LSTM, self).__init__()
        self.num_class = num_class
        self.embedding_dim = embedding_dim
        self.device = device
        self.segment_len = segment_len
        self.rnn_dim = 100
        # if bi-direction
        self.rnn_directions = 2 if bidirectional else 1

        # 256 is 'gg', will be set [0,0..0]
        self.byte_embed = nn.Embedding(257, self.embedding_dim, padding_idx=256)
        # to one-hot
        self.byte_embed.requires_grad = True

        self.rnn1 = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.rnn_dim,
                            batch_first=True, dropout=dropout_rate,
                            bidirectional=(self.rnn_directions == 2))
        self.rnn2 = nn.LSTM(self.rnn_directions * self.rnn_dim, self.rnn_dim,
                            batch_first=True, dropout=dropout_rate,
                            bidirectional=(self.rnn_directions == 2))

        # consider other initialization
        self.hc1 = nn.parameter.Parameter(torch.randn(self.rnn_dim, 1))
        self.hc2 = nn.parameter.Parameter(torch.randn(self.rnn_dim, 1))

        self.fc1 = nn.Linear(self.rnn_dim * self.rnn_directions,
                             self.rnn_dim)
        self.fc2 = nn.Linear(self.rnn_dim * self.rnn_directions,
                             self.rnn_dim)
        self.fc3 = nn.Linear(self.rnn_dim * self.rnn_directions,
                             self.num_class)

        self.dropout = nn.Dropout(0.5)

    '''
    x: b * l * 8
    lengths: (b,), every batch's length
    '''

    def forward(self, x):
        batch_size = x.size(0)
        x = self.byte_embed(x)  # b * l * 8 * 257

        out1, (h_n, c_n) = self.rnn1(x.view(-1, self.segment_len, self.embedding_dim))
        # bl * 8 * 100
        h = torch.tanh(self.fc1(out1))
        weights = (torch.matmul(h, self.hc1)).view(-1, self.segment_len)
        weights = F.softmax(weights, dim=1)
        weights = weights.view(-1, 1, self.segment_len)
        # b * l * 200
        out2 = torch.matmul(weights, out1).view(
            batch_size, -1, self.rnn_dim * self.rnn_directions)

        out3, (h1_n, h2_n) = self.rnn2(out2)  # out3: b * l * 200
        h2 = torch.tanh(self.fc2(out3))
        weights2 = F.softmax((torch.matmul(h2, self.hc2)).view(
            batch_size, -1), dim=1).view(batch_size, 1, -1)
        out4 = torch.matmul(weights2, out3).view(
            batch_size, self.rnn_dim * self.rnn_directions)

        out = self.dropout(self.fc3(out4))
        return out


class FocalLoss(nn.Module):
    def __init__(self, class_num, device, alpha=None, gamma=2,
                 size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = nn.parameter.Parameter(torch.ones(class_num, 1), requires_grad=False)
        else:
            self.alpha = nn.parameter.Parameter(alpha, requires_grad=False)
        self.device = device
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=-1)

        class_mask = inputs.data.new(N, C).fill_(0)
        ids = targets.view(-1, 1)

        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        # if inputs.is_cuda and not self.alpha.is_cuda:
        #     self.alpha = self.alpha.to(self.device)
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class KDLoss(nn.Module):
    
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    def __init__(self, alpha, temperature, args):
        self.alpha = alpha
        self.T = temperature
        if args.focal:
            self.supervised_loss = nn.CrossEntropyLoss()
        else:
            self.supervised_loss = FocalLoss(args.num_classes, args.device, args.gamma, True)


    def forward(self, outputs, labels, teacher_outputs):

        KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/self.T, dim=1),
                                F.softmax(teacher_outputs/self.T, dim=1)) * \
                                    (self.alpha * self.T * self.T) + \
                self.supervised_loss(outputs, labels) * (1. - self.alpha)

        return KD_loss