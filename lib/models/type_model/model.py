import sys

sys.path.append("..")

import numpy as np
import torch
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel as BM

from torch import nn, optim
from torch.autograd import Variable
from tqdm import tqdm

bert_name = 'bert_base_chinese'
bert_path = '../bert-base-chinese'
bert_cache_path = './tmp/bert'

bert_outsize = 768
linear_model = 0
learning_rate = 0.1
epoch_num = 100

use_gpu = torch.cuda.is_available()
print("use_gpu: " + str(use_gpu))


def tokensfit(ids, finallen):
    nowlen = len(ids)
    if nowlen < finallen:
        ids += [116] * (finallen - nowlen)
    elif nowlen > finallen:
        ids = ids[:finallen]
    return ids


class LinearModel(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(LinearModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, n_hidden_1),
            nn.Dropout(0.1))
        self.layer2 = nn.Sequential(
            nn.BatchNorm1d(n_hidden_1),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.Dropout(0.1))
        self.layer3 = nn.Sequential(
            nn.BatchNorm1d(n_hidden_2),
            nn.Linear(n_hidden_2, out_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = torch.tanh(x)
        x = self.layer2(x)
        x = torch.tanh(x)
        x = self.layer3(x)
        return x
