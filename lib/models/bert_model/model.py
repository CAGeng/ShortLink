import sys
sys.path.append("..")

import numpy as np
import torch
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel as BM

from torch import nn,optim
from torch.autograd import Variable

bert_name = 'bert-base-chinese'
bert_path = './lib/models/bert_base_chinese'
bert_cache_path = './tmp'
bert_outsize = 768
linear_model = 0
learning_rate = 0.1
epoch_num = 100

def tokensfit(ids, finallen):
    nowlen = len(ids)
    if nowlen < finallen:
        ids += [116] * (finallen - nowlen)
    elif nowlen > finallen:
        ids = ids[:finallen]
    return ids

class BertModel():
    def __init__(self, use_gpu):
        self.model = BM.from_pretrained(bert_path)
        # self.model.load_state_dict(torch.load("./checkpoint/bert_model/model_temp4.pth"))
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        self.use_gpu = use_gpu
        if self.use_gpu:
            self.model = self.model.cuda()

    def batch_forward(self, text):
        # print(text)
        tokenized_text = [self.tokenizer.tokenize(i) for i in text]                 #将句子分割成一个个token，即一个个汉字和分隔符
        input_ids = [tokensfit(self.tokenizer.convert_tokens_to_ids(i), finallen=128) for i in tokenized_text]  #把每个token转换成对应的索引
        input_ids = torch.LongTensor(input_ids)

        if self.use_gpu:
            input_ids = input_ids.cuda()

        # self.model.eval()
        outputs = self.model(input_ids)
        prediction_scores = outputs[0]
        sample = prediction_scores.detach().cpu().numpy()
        # sample = prediction_scores
        return sample

    def batch_forward_grad(self, text):  #for backward
        # print(text)
        tokenized_text = [self.tokenizer.tokenize(i) for i in text]                 #将句子分割成一个个token，即一个个汉字和分隔符
        input_ids = [tokensfit(self.tokenizer.convert_tokens_to_ids(i), finallen=128) for i in tokenized_text]  #把每个token转换成对应的索引
        input_ids = torch.LongTensor(input_ids)

        if self.use_gpu:
            input_ids = input_ids.cuda()

        # self.model.eval()
        outputs = self.model(input_ids)
        prediction_scores = outputs[0]
        # sample = prediction_scores.detach().cpu().numpy()
        sample = prediction_scores
        return sample