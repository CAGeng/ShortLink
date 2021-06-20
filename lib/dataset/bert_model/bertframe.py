import torch
import numpy as np
from tqdm import tqdm

from lib.models.bert_model.model import BertModel as bert_model

class BertFrame:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        self.bert_model = bert_model(use_gpu)
        # self.bert_model.model.eval()

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
            # ele_offset = []

            ele_offset.append(0)
            ele_output = ele_output[ele_offset]
            ele_output = ele_output.flatten()
            batch_output.append(ele_output)
        batch_output = np.array(batch_output)
        batch_output = torch.tensor(batch_output)
        return batch_output

    def bert_batch_forward2(self, text, beg_en):
        '''
        text: ['啊哈'，'哦吼']
        beg_en: [[0, 2], [0, 1]]
        '''
        x = self.bert_model.batch_forward_grad(text) 
        batch_output = []
        for i in range(x.shape[0]):
            ele_output = x[i]
            ele_offset = beg_en[i]
            ele_offset.append(0)
            ele_output = ele_output[ele_offset]
            ele_output = ele_output.flatten()
            batch_output.append(ele_output)
        # batch_output = np.array(batch_output)
        batch_output = [x.unsqueeze(0) for x  in batch_output]
        batch_output = torch.cat(batch_output,dim=0)
        return batch_output

    def bert_process_data(self, train_data,need_tqdm=1):
        if need_tqdm:
            print("Bert_forwarding")
        train_data_process = []
        if need_tqdm == 1:
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
                    linear_input = self.bert_batch_forward(textlist, begenlist)
                if self.use_gpu:
                    linear_input = linear_input.cpu()
                train_data_process.append(linear_input)
        else:
            for i in range(len(train_data)):
                batch_data = train_data[i]
                textlist = []
                begenlist = []
                for j in range(len(batch_data)):
                    textlist.append(batch_data[j]['text'])
                    beg = batch_data[j]['beg'] + 1
                    en = batch_data[j]['en'] + 1
                    begenlist.append([beg,en])
                with torch.no_grad():
                    linear_input = self.bert_batch_forward(textlist, begenlist)
                if self.use_gpu:
                    linear_input = linear_input.cpu()
                train_data_process.append(linear_input)
        return train_data_process