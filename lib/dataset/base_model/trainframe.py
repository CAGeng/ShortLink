import pandas as pd
import numpy as np
import torch
import random

from lib.utils import entity_recognition
from lib.utils import generate_entity_text, generate_text, get_name_dic

from lib.utils import *
from lib.models.bert_model.model import BertModel
from lib.models.base_model.model import LinearModel

class TrainFrame:
    def __init__(self, use_gpu, in_dim, n_hidden_1, n_hidden_2, out_dim):
        self.use_gpu = use_gpu

        self.model = LinearModel(in_dim, n_hidden_1, n_hidden_2, out_dim)
        self.bert_model = BertModel(use_gpu)
        if use_gpu:
            self.model = self.model.cuda()

        self.KEY_DIC = {}
        self.KB = pd.DataFrame()
        self.traindf = pd.DataFrame()


    def readdata(self):
        print("-----------------------读入知识库、训练集、NameDic--------------------------------------")
        self.KB = pd.read_json("./data/baidu/kb.json", lines=True)
        # KB = KB.set_index('subject_id')
        self.traindf = pd.read_json("./data/baidu/train.json", lines=True)
        # self.traindf = self.traindf.head(300)
        self.name_dict = get_name_dic(self.KB)
        # self.name_dict = load_obj("name_dict")

    def create_pos_trainset(self):
        KB_id = self.KB.set_index('subject_id')
        # traindf = pd.read_json("../input/train.json",lines=True)
        train_data = []
        train_label = []
        for i in tqdm(range(self.traindf.shape[0])):
            mention_text = self.traindf['text'][i]
            mention_data = self.traindf['mention_data'][i]
            for data in mention_data:
                entity_id = data['kb_id']
                if entity_id.startswith("NIL"):
                    continue
                else:
                    entity_id = int(entity_id)
                entity = KB_id.loc[entity_id, :]
                entity_text = generate_entity_text(entity['data'], entity['type'])
                text = generate_text(mention_text, entity_text)
                begpos = int(data['offset'])
                endpos = begpos + len(data['mention']) -1

                train_data.append({'text': text,
                                   'beg': begpos,
                                   'en': endpos})
                train_label.append(1)
        # train_label = torch.tensor(train_label)
        return train_data, train_label

    def create_neg_trainset(self):
        KB_id = self.KB.set_index('subject_id')
        # traindf = pd.read_json("../input/train.json",lines=True)
        train_data = []
        train_label = []

        for i in tqdm(range(self.traindf.shape[0])):
            mention_text = self.traindf['text'][i]
            mention_data = self.traindf['mention_data'][i]
            for data in mention_data:
                mention_name = data['mention']
                entity_id_pos = data['kb_id']

                entities = entity_recognition(mention_name, KB_id, self.name_dict)
                if entities is None:
                    continue
                if entities.shape[0] == 1:
                    continue
                entity_id_choose = random.sample(list(entities.index), 2)
                if entity_id_choose[0] == entity_id_pos:
                    entity_id = entity_id_choose[1]
                else:
                    entity_id = entity_id_choose[0]
                entity = KB_id.loc[entity_id, :]
                entity_text = generate_entity_text(entity['data'], entity['type'])

                text = generate_text(mention_text, entity_text)
                begpos = int(data['offset'])
                endpos = begpos + len(data['mention']) - 1

                train_data.append({'text': text,
                                   'beg': begpos,
                                   'en': endpos})
                train_label.append(0)
        # train_label = torch.tensor(train_label)
        return train_data, train_label

    def mix_pos_neg(self, posdata, negdata, poslabel, neglabel):
        data = posdata + negdata
        label = poslabel + neglabel
        return data, label

    def bert_batch_forward(self, text, beg_en):
        '''
        text: ['啊哈'，'哦吼']
        beg_en: [[0, 2], [0, 1]]
        '''
        x = self.bert_model.batch_forward(text)  # numpy([batchsize, len, 768])
        batch_output = []
        for i in range(x.shape[0]):
            ele_output = x[i]
            ele_offset = beg_en[i]
            ele_offset.append(0)
            ele_output = ele_output[ele_offset]
            ele_output = ele_output.flatten()
            batch_output.append(ele_output)
        batch_output = np.array(batch_output)
        batch_output = torch.tensor(batch_output)
        return batch_output

    def bert_process_data(self, train_data):
        print("Bert_forwarding")
        train_data_process = []
        if self.use_gpu:
            self.bert_model.cuda()
        self.bert_model.eval()
        for i in tqdm(range(len(train_data))):
            batch_data = train_data[i]
            textlist = []
            begenlist = []
            for j in range(len(batch_data)):
                textlist.append(batch_data[j]['text'])
                beg = batch_data[j]['beg'] + 1
                en = batch_data[j]['en'] + 1
                begenlist.append([beg,en])
            with torch.no_grad():
                linear_input = self.bert_model.bert_batch_forward(textlist, begenlist)
            if self.use_gpu:
                linear_input = linear_input.cpu()
            train_data_process.append(linear_input)
        return train_data_process

    def reget_trainset(self):
        self.readdata()
        print("-----------------------重新获得训练集--------------------------------------")
        train_data_pos, train_label_pos = self.create_pos_trainset()
        train_data_neg, train_label_neg = self.create_neg_trainset()
        train_data, train_label = self.mix_pos_neg(train_data_pos, train_data_neg, train_label_pos, train_label_neg)
        train_label = torch.tensor(train_label)
        return train_data, train_label






