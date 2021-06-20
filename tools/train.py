import sys
sys.path.append(".")

import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from lib.utils import *

from lib.dataset.bert_model.bertframe import BertFrame as bert_frame

from lib.dataset.base_model.trainframe import TrainFrame as base_tf
from lib.dataset.type_model.trainframe import TrainFrame as type_tf
from lib.dataset.NILtype_model.trainframe import TrainFrame as NILtype_tf

KB = pd.DataFrame()
traindf = pd.DataFrame()
Name_Dic = {}
dataset_path = "./data/baidu"
bert_outsize = 768

class Train:
    def __init__(self, trainframe, model_name, learning_rate = 0.1, use_gpu=True, re_bert = True, epoch_num = 100):
        self.trainframe = trainframe
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu
        self.model_name = model_name
        self.re_bert = re_bert
        self.epoch_num = epoch_num

        self.bert_frame = bert_frame(use_gpu)

    def divide_batch(self, train_data, train_label, batch_size=32):
        data_batchs = []
        label_batchs = []
        print("-----------------------划分batch--------------------------------------")
        ind = 0
        while ind + batch_size < len(train_data):
            databatch = train_data[ind:ind + batch_size]
            labelbatch = train_label[ind:ind + batch_size]
            data_batchs.append(databatch)
            label_batchs.append(labelbatch)
            ind += batch_size
        # data_batchs.append(train_data[ind:])
        # label_batchs.append(train_label[ind:])
        return data_batchs, label_batchs

    def divide_batch_tensor(self, train_data, train_label, batch_size=32):
        data_batchs = []
        label_batchs = []
        print("-----------------------划分batch--------------------------------------")
        ind = 0
        while ind + batch_size <= train_data.shape[0]:
            databatch = train_data[ind:ind + batch_size]
            labelbatch = train_label[ind:ind + batch_size]
            data_batchs.append(databatch)
            label_batchs.append(labelbatch)
            ind += batch_size
        # data_batchs.append(train_data[ind:])
        # label_batchs.append(train_label[ind:])
        return data_batchs, label_batchs

    def shuffle(self, data, label):
        ind = [i for i in range(data.shape[0])]
        random.shuffle(ind)
        # print(data[1])
        data = data[ind]
        label = label[ind]
        return data, label
    
    def shuffle2(self, data, label):
        ind = [i for i in range(len(data))]
        random.shuffle(ind)
        # print(data[1])
        data = [data[ind[i]] for i in range(len(ind))]
        label = torch.tensor([label[ind[i]] for i in range(len(ind))])
        return data, label

    def batch_train(self, train_data, train_label, need_bert=0):
        '''
        train_data: [
            [{ 'text':'', 'beg':0, 'en':1}, {}]    ---batch
        ]
        train_label: [
            [1,0,0,1]               ---batch
        ]
        '''
        # 文本通过bert
        if need_bert == 1:
            train_data = self.bert_frame.bert_process_data(train_data)
            # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
        if self.use_gpu:
            criterion = criterion.cuda()
        optimizer = optim.SGD(self.trainframe.model.parameters(), lr=0.1, weight_decay=0.001)

        self.trainframe.model.train()
        loss_list = []
        for epoch in range(self.epoch_num):
            print("epoch:{}".format(epoch))
            epoch_loss = []
            for i in tqdm(range(len(train_data))):
                batch_label = train_label[i]

                linear_input = train_data[i]
                linear_input = torch.tensor(linear_input, requires_grad=True)
                if self.use_gpu:
                    batch_label = batch_label.cuda()
                    linear_input = linear_input.cuda()

                out = self.trainframe.model(linear_input)

                loss = criterion(out, batch_label)
                epoch_loss.append(loss.data.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss = np.array(epoch_loss)
            # print(epoch_loss)
            mean_loss = epoch_loss.mean()
            print('epoch: {}, loss: {:.4}'.format(epoch, mean_loss))
            loss_list.append(mean_loss)

            save_T = 25
            if epoch % save_T == 0 and epoch != 0:
                temp = epoch // save_T
                model_name = "model_temp" + str(temp) + ".pth"
                torch.save(self.trainframe.model.state_dict(), os.path.join('./checkpoint', self.model_name, model_name))
                print(os.path.join('./checkpoint', self.model_name, model_name))
        return loss_list

    def batch_train_bert(self, train_data, train_label, need_bert=0):
        '''
        train_data: [
            [{ 'text':'', 'beg':0, 'en':1}, {}]    ---batch
        ]
        train_label: [
            [1,0,0,1]               ---batch
        ]
        '''
        # 文本通过bert
        if need_bert == 1:
            train_data = self.bert_frame.bert_process_data(train_data)
            # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
        if self.use_gpu:
            criterion = criterion.cuda()

        self.trainframe.model.train()
        self.bert_frame.bert_model.model.train()

        # standard_data = load_obj(os.path.join("output", self.model_name, "processed_train_data.pkl"))
        # standard_label = load_obj(os.path.join("output", self.model_name, "train_label.pkl"))
        # raw_train_data, raw_train_label = self.trainframe.reget_trainset()

        loss_list = []
        for epoch in range(self.epoch_num):
            jjj = 1
            optimizer = optim.SGD(self.bert_frame.bert_model.model.parameters(), lr=0.00001 / jjj)
            optimizer2 = optim.SGD(self.trainframe.model.parameters(), lr=0.005 / jjj)
            jjj *= 2
            print("epoch:{}".format(epoch))
            epoch_loss = []
            for i in range(len(train_data)):
                batch_label = train_label[i]

                batch_data = train_data[i]
                textlist = []
                begenlist = []
                for j in range(len(batch_data)):
                    textlist.append(batch_data[j]['text'])
                    beg = batch_data[j]['beg'] + 1
                    en = batch_data[j]['en'] + 1
                    begenlist.append([beg,en])
                
                linear_input = self.bert_frame.bert_batch_forward2(textlist, begenlist)

                # linear_input = train_data[i]
                # linear_input = torch.tensor(linear_input, requires_grad=True)
                # print(len(texts))
                
                # linear_input = self.bert_frame.bert_model.batch_forward_grad(texts)
                # linear_input = linear_input[:,0,:]
                # print('before')
                # print(linear_input[0,0])
                # linear_input = torch.tensor(linear_input, requires_grad=True)
                if self.use_gpu:
                    batch_label = batch_label.cuda()
                    linear_input = linear_input.cuda()

                out = self.trainframe.model(linear_input)

                loss = criterion(out, batch_label)
                epoch_loss.append(loss.data.item())
                optimizer.zero_grad()
                optimizer2.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer2.step()
                # linear_input = self.bert_frame.bert_model.batch_forward(texts)
                # linear_input = linear_input[:,0,:]
                # linear_input = self.bert_frame.bert_batch_forward(textlist, begenlist)
                # print('after')
                # print("output: {}".format(out.argmax(dim=1)))
                # print("label: {}".format(batch_label))
                print(np.asarray(epoch_loss).mean())
                # print("--")

            epoch_loss = np.array(epoch_loss)
            # print(epoch_loss)
            mean_loss = epoch_loss.mean()
            print('epoch: {}, loss: {:.4}'.format(epoch, mean_loss))
            loss_list.append(mean_loss)

            save_T = 1
            if epoch % save_T == 0 and epoch != 0:
                temp = epoch // save_T
                model_name = "model_temp" + str(temp) + ".pth"
                torch.save(self.trainframe.model.state_dict(), os.path.join('./checkpoint', self.model_name, model_name))
                torch.save(self.bert_frame.bert_model.model.state_dict(), os.path.join('./checkpoint', 'bert_model', model_name))
                self.bert_frame.bert_model.model.save_pretrained(os.path.join('./checkpoint', self.model_name, 'bert_model'))
                print(os.path.join('./checkpoint', self.model_name, model_name))
                
        return loss_list

    def train(self):
        print(os.listdir("./"))

        if self.re_bert:
            train_data, train_label = self.trainframe.reget_trainset()  # list, tensor
            print('trainset:')
            print(len(train_data))
            train_data, _ = self.divide_batch(train_data, train_label, batch_size=768) #768
            train_data = self.bert_frame.bert_process_data(train_data)
            train_data = torch.cat(train_data, dim=0)  # tensor(未划分)
            save_obj(train_data, os.path.join("output", self.model_name, "processed_train_data.pkl"))
            save_obj(train_label, os.path.join("output", self.model_name, "train_label.pkl"))
        else:
            print("-----------------------获得缓存的数据集--------------------------------------")
            train_data = load_obj(os.path.join("output", self.model_name, "processed_train_data.pkl"))
            train_label = load_obj(os.path.join("output", self.model_name, "train_label.pkl"))

        print(train_data[0])
        print(train_data[0].shape)
        print(train_data.shape)

        # train_data = train_data[: , :2 * 768]
        train_data, train_label = self.shuffle(train_data, train_label)  # tensor,tensor 已经随机打乱
        train_data, train_label = self.divide_batch_tensor(train_data, train_label, batch_size=1024) # 1024  # tensor_list,tensor_list
        loss_list = self.batch_train(train_data, train_label, need_bert=0)
        torch.save(self.trainframe.model.state_dict(), os.path.join('./checkpoint', self.model_name + '.pth'))
        plot_curve(loss_list)

    def train_bert(self):
        train_data, train_label = self.trainframe.reget_trainset()  # list, tensor
        train_data, train_label = self.shuffle2(train_data, train_label)
        # train_data = load_obj(os.path.join("output", "base_model", "processed_train_data.pkl"))
        # train_label = load_obj(os.path.join("output", "base_model", "train_label.pkl"))
        # train_data, train_label = self.shuffle(train_data, train_label)  # tensor,tensor 已经随机打乱
        # train_data, train_label = self.divide_batch_tensor(train_data, train_label, batch_size=1024) # 1024  # tensor_list,
        print('trainset:')
        print(len(train_data))
        train_data, train_label = self.divide_batch(train_data, train_label, batch_size=64) #768
        loss_list = self.batch_train_bert(train_data, train_label)
        plot_curve(loss_list)

if __name__ == "__main__":
    base_tf = base_tf(torch.cuda.is_available(), bert_outsize * 3, 128, 36, 2)
    base_train = Train(base_tf, "base_model", re_bert=False, epoch_num=500)
    base_train.train()

    type_tf = type_tf(torch.cuda.is_available(), bert_outsize * 3, 512, 128, 24)
    # type_tf = type_tf(torch.cuda.is_available(), bert_outsize * 3, 128, 32, 24)
    type_train = Train(type_tf, "type_model", re_bert=False)
    type_train.train()

    NILtype_tf = NILtype_tf(torch.cuda.is_available(), bert_outsize * 3, 128, 32, 24)
    type_train = Train(NILtype_tf, "NILtype_model", re_bert=True, epoch_num=150)
    type_train.train()

    # standard_data = load_obj(os.path.join("output", "base_model", "processed_train_data.pkl"))
    # standard_label = load_obj(os.path.join("output", "base_model", "train_label.pkl"))
    # print(len(standard_data))
    # print(len(standard_label))

    # while True:
    #     pass


    # use_gpu = torch.cuda.is_available()
    # bert_tf_ = base_tf(use_gpu, bert_outsize * 3, 128, 36, 2)
    # bert_tf_.model.load_state_dict(torch.load("./models/base_model.pth"))
    # bert_train = Train(bert_tf_, "base_model", use_gpu=use_gpu, epoch_num=10)
    # bert_train.train_bert()
