from base64 import encode
import sys
sys.path.append(".")

import argparse
import os
import numpy as np
import pandas as pd
import torch
import copy
import math
import json
from tqdm import tqdm

from lib.utils import *

from lib.vote.vote import Vote as vote
from lib.dataset.bert_model.bertframe import BertFrame as bert_frame
from lib.models.base_model.model import LinearModel as base_model
from lib.models.type_model.model import LinearModel as type_model
from lib.models.type_model.model import LinearModel as NILtype_model

# VOTE_RATE = [3, 1, 0.1, 1]
bert_outsize = 768

class Evaluate:
    def __init__(self, re_bert=False, eval=True, type_sel=-1):
        self.KB = pd.DataFrame()
        self.datadf = pd.DataFrame()

        self.name_dict = pd.DataFrame()
        self.ENTITY_CON = {}
        self.COH_DIC = {}
        self.ENTITY_CON_TOTAL = {}
        self.POPULARITY_DIC = {}

        self.TYPE_LIST = ['Person', 'Work', 'Other', 'Location', 'Culture', 'VirtualThings', 'Organization', 'Game',
                          'Biological', 'Software', 'Website', 'Food', 'Brand', 'Medicine', 'Event',
                          'Natural&Geography', 'Time&Calendar', 'Vehicle', 'Disease&Symptom', 'Education', 'Awards',
                          'Constellation', 'Diagnosis&Treatment', 'Law&Regulation']
        self.TYPE_DIC = {}

        self.VOTE_RATE = [3, 1, 0.6, 1.2]
        self.VOTE_RATE_CHECKNIL = [0, 0, 0, 1]
        self.THRESHOLD = 0.05

        self.re_bert = re_bert
        self.eval = eval
        self.type_sel = type_sel
        self.use_gpu = torch.cuda.is_available()
        self.bert_frame = bert_frame(self.use_gpu)
        self.base_model = base_model(bert_outsize * 3, 128, 36, 2)
        self.type_model = type_model(bert_outsize * 3, 512, 128, 24)
        # self.NILtype_model = NILtype_model(bert_outsize * 3, 512, 128, 24)
        self.NILtype_model = NILtype_model(bert_outsize * 3, 128, 32, 24)
        self.vote = vote(self.POPULARITY_DIC, self.KB, self.datadf)

    def readdata(self):
        print("-----------------------读入--------------------------------------")
        # 基础
        self.KB = pd.read_json("./data/baidu/kb.json", lines=True)
        if self.eval:
            self.datadf = pd.read_json("./data/baidu/dev.json", lines=True)
        else:
            self.datadf = pd.read_json("./data/baidu/test.json", lines=True)
        # self.datadf = self.datadf.head(300)

        # 统计数据
        self.name_dict = load_obj("./models/dict/name_dict.pkl")
        self.ENTITY_CON = load_obj("./models/dict/ENTITY_CON.pkl")
        self.ENTITY_CON_TOTAL = load_obj("./models/dict/ENTITY_CON_TOTAL.pkl")
        self.COH_DIC = load_obj("./models/dict/COH_DIC.pkl")
        self.POPULARITY_DIC = load_obj("./models/dict/POPULARITY_DIC.pkl")

        # 类型（上位概念）
        for i in range(len(self.TYPE_LIST)):
            self.TYPE_DIC[self.TYPE_LIST[i]] = i

    def create_test_data(self, KB, datadf, name_dict):
        print("-----------------------获得测试集--------------------------------------")
        KB_id = self.KB.set_index('subject_id')
        dataset_1 = []
        dataset_2 = []
        ind = 0
        real_entity_id_list = []
        sentence_list = []
        for i in tqdm(range(datadf.shape[0])):
            # sentence
            mention_text = datadf['text'][i]
            mention_data = datadf['mention_data'][i]
            Sentence = []
            for data in mention_data:
                m = data['mention']
                if self.eval:
                    real_entity_id_list.append(data['kb_id'])
                Mention = []
                Mention.append(-1)

                entities = entity_recognition(m, KB_id, name_dict)
                if entities is None:
            
                    begpos = int(data['offset'])
                    endpos = begpos + len(data['mention']) - 1
                    # # model1 的输入,这里用来凑数
                    # dataset_1.append({'text': mention_text,
                    #                   'beg': begpos,
                    #                   'en': endpos})
                    # # model2 的输入
                    # dataset_2.append({'text': mention_text,
                    #                   'beg': begpos - 1,
                    #                   'en': endpos - 1})

                    Mention[0] = [-38,{'text': "[CLS]" + mention_text,
                                      'beg': begpos ,
                                      'en': endpos }]
                    # ind += 1
                    Sentence.append(Mention)
                    continue

                entities_len = entities.shape[0]
                for i in range(entities_len):
                    e = str(entities.index[i])
                    entity = entities.iloc[i, :]
                    entity_text = generate_entity_text(entity['data'], entity['type'])
                    text = generate_text(mention_text, entity_text)
                    begpos = int(data['offset'])
                    endpos = begpos + len(data['mention']) - 1
                    # model1 的输入
                    dataset_1.append({'text': text,
                                      'beg': begpos,
                                      'en': endpos})
                    # model2 的输入
                    dataset_2.append({'text': "[CLS]" + mention_text,
                                      'beg': begpos ,
                                      'en': endpos })

                    Mention[0] = [-38,{'text': "[CLS]" + mention_text,
                                      'beg': begpos ,
                                      'en': endpos }]

                    Mention.append([m, e, ind])
                    ind += 1
                Sentence.append(Mention)
            sentence_list.append(Sentence)
        return dataset_1, dataset_2, sentence_list, real_entity_id_list

    def divide_batch_data(self, data, batch_size=32):
        data_batchs = []
        print("-----------------------划分batch--------------------------------------")
        ind = 0
        while ind + batch_size <= len(data):
            databatch = data[ind:ind + batch_size]
            data_batchs.append(databatch)
            ind += batch_size

        if ind < len(data):
            data_batchs.append(data[ind:])
        return data_batchs

    def divide_batch_tensor_data(self, train_data, batch_size=32):
        data_batchs = []
        print("-----------------------划分batch--------------------------------------")
        ind = 0
        while ind + batch_size <= train_data.shape[0]:
            databatch = train_data[ind:ind + batch_size]
            data_batchs.append(databatch)
            ind += batch_size

        if ind < train_data.shape[0]:
            data_batchs.append(train_data[ind:])
        return data_batchs

    def batch_forward(self, model, x,need_tqdm=1):
        outs = []
        if self.use_gpu:
            model.cuda()
        model.eval()
        if need_tqdm == 1:
            for i in tqdm(range(len(x))):
                linear_input = x[i]
                linear_input = torch.tensor(linear_input, requires_grad=False)
                if self.use_gpu:
                    linear_input = linear_input.cuda()
                with torch.no_grad():
                    out = model(linear_input)
                out = torch.softmax(out, dim=1)
                outs.append(out)
        else:
            for i in range(len(x)):
                linear_input = x[i]
                linear_input = torch.tensor(linear_input, requires_grad=False)
                if self.use_gpu:
                    linear_input = linear_input.cuda()
                with torch.no_grad():
                    out = model(linear_input)
                out = torch.softmax(out, dim=1)
                outs.append(out)
        return outs

    def model_forward(self, model, dataset, bert_batchsize=64, linear_batchsize=4096,name="1"):
        x = self.divide_batch_data(dataset, batch_size=bert_batchsize)
        x = self.bert_frame.bert_process_data(x)
        x = torch.cat(x, dim=0)  # tensor(未划分)

        save_obj(x, "./output/bertout/" + name + ".pkl")

        # x = load_obj("./output/bertout/" + name + ".pkl")

        x = self.divide_batch_tensor_data(x, batch_size=linear_batchsize)  # tensor_list
        x = self.batch_forward(model, x)
        x = torch.cat(x, dim=0)  # tensor(未划分)
        return x

    def cal_maxcoh(self, e, elist):
        if len(elist) == 1:
            return 0
        maxcoh = 0
        for i in range(1, len(elist)):
            e2 = elist[i][1]
            tmp = self.vote.get_coh_e1e2(e, e2, self.ENTITY_CON, self.ENTITY_CON_TOTAL, self.COH_DIC)
            maxcoh = max(maxcoh, tmp)
        return maxcoh

    def get_predict(self, sentence_list, model1_out, model2_out,NILmodel):
        KB_id = self.KB.set_index('subject_id')
        Ans_List = []
        print("-----------------------预测--------------------------------------")

        # print(sentence_list[0])
        # print(sentence_list[0][0])

        # while True:
        #     pass

        for Sentence in tqdm(sentence_list):
            for i in range(len(Sentence)):
                Mention = Sentence[i]
                if len(Mention) == 1:
                    # ind = Mention[0][0]
                    # type_softmax = model2_out[ind]
                    # typeid = type_softmax.argmax(dim=0)

                    bert_input = Mention[0][1]
                    x = self.bert_frame.bert_process_data([[bert_input]],need_tqdm=0)
                    # print(x[0].shape)
                    # x = [x[0][:,:2*768]]
                    # print(x[0].shape)
                    x = self.batch_forward(NILmodel, x,need_tqdm=0)
                    # x = torch.cat(x, dim=0)
                    x = x[0]
                    NILtype_softmax = x.squeeze()
                    typeid = NILtype_softmax.argmax(dim=0)
                    
                    typename = self.TYPE_LIST[typeid]
                    if not self.eval:
                        Ans_List.append('NIL_' + typename)
                    else:
                        Ans_List.append({
                            "answer": 'NIL_' + typename,
                            "nil_score": NIL_scores,
                            "score": scores,
                            "mention": Mention})
                    continue

                # 计算实体集评分
                scores = np.zeros((len(Mention) - 1, 4))

                for ek in range(1, len(Mention)):
                    m = Mention[ek][0]
                    e = Mention[ek][1]
                    ind = Mention[ek][2]

                    # 计算Coh
                    score = 0
                    con_temp = 0
                    for j in range(len(Sentence)):
                        if i == j:
                            continue
                        score += self.cal_maxcoh(e, Sentence[j])
                        con_temp += 1
                    if con_temp == 0:
                        scores[ek - 1, 0] = 0
                    else:
                        scores[ek - 1, 0] = score / con_temp

                    # Pop
                    if m not in self.POPULARITY_DIC.keys():
                        score = 0
                    elif e not in self.POPULARITY_DIC[m].keys():
                        score = 0
                    else:
                        score = self.POPULARITY_DIC[m][e]
                    scores[ek - 1, 1] = score

                    # type
                    e_type = KB_id.loc[int(e), 'type']
                    if e_type not in self.TYPE_DIC.keys():
                        e_typeid = 2  # 'Other'
                    else:
                        e_typeid = self.TYPE_DIC[e_type]
                    scores[ek - 1, 2] = model2_out[ind][e_typeid]

                    # Relativity
                    scores[ek - 1, 3] = model1_out[ind][1]

                # 判断是否为NIL，若是，predict为上位概念
                def vote_score_nil(x):
                    ret = self.VOTE_RATE_CHECKNIL[0] * x[0] + self.VOTE_RATE_CHECKNIL[1] * x[1] + \
                          self.VOTE_RATE_CHECKNIL[2] * x[2] + self.VOTE_RATE_CHECKNIL[3] * x[3]
                    ret = ret / (self.VOTE_RATE_CHECKNIL[0] + self.VOTE_RATE_CHECKNIL[1] + \
                                 self.VOTE_RATE_CHECKNIL[2] + self.VOTE_RATE_CHECKNIL[3])
                    return ret

                NIL_scores = [vote_score_nil(scores[x]) for x in range(len(Mention) - 1)]
                max_NIL_scores = max(NIL_scores)
                # print(max_NIL_scores)
                if max_NIL_scores < self.THRESHOLD:
                    # print(max_NIL_scores)
                    # print(Mention)
                    bert_input = Mention[0][1]
                    x = self.bert_frame.bert_process_data([[bert_input]],need_tqdm=0)
                    # x = [x[0][:,:2*768]]
                    x = self.batch_forward(NILmodel, x,need_tqdm=0)
                    # x = torch.cat(x, dim=0)
                    x = x[0]
                    NILtype_softmax = x.squeeze()
                    typeid = NILtype_softmax.argmax(dim=0)
                    
                    typename = self.TYPE_LIST[typeid]
                    if not self.eval:
                        Ans_List.append('NIL_' + typename)
                    else:
                        Ans_List.append({
                            "answer": 'NIL_' + typename,
                            "nil_score": NIL_scores,
                            "score": scores,
                            "mention": Mention})
                    continue

                # 排序
                index = [i for i in range(len(Mention) - 1)]

                def vote_score(x):
                    ret = self.VOTE_RATE[0] * x[0] + self.VOTE_RATE[1] * x[1] + \
                          self.VOTE_RATE[2] * x[2] + self.VOTE_RATE[3] * x[3]
                    return ret

                index = sorted(index, key=lambda x: vote_score(scores[x]), reverse=True)
                best_index = index[0]

                if not self.eval:
                    Ans_List.append(Mention[best_index + 1][1])
                else:
                    Ans_List.append({
                        "answer": Mention[best_index + 1][1],
                        "nil_score": NIL_scores,
                        "score": scores,
                        "mention": Mention})

        return Ans_List

    def check_predict(self, predict_list_all, real_list):
        total = 0
        correct = 0
        NIL_total = 0
        NIL_correct = 0
        Nocall_num = 0
        Wrongcall_num = 0

        predict_list = []
        for i in range(len(predict_list_all)):
            # predict_list = [x["answer"] for x in predict_list_all]
            x = predict_list_all[i]
            # print(x)
            # print(x['answer'])
            # print(i)
            predict_list.append(x['answer'])

        wrong_list = []
        nil_total = 0
        nil_wrong = 0
        nil_person_total = 0
        nil_person_wrong = 0
        for i in range(len(predict_list)):
            # if not (real_list[i].startswith('NIL') and predict_list[i].startswith('NIL')):
            #     continue

            if real_list[i].startswith('NIL'):
                NIL_total += 1
                if predict_list[i] == real_list[i]:
                    NIL_correct += 1
                else:
                    # print(predict_list[i],real_list[i])
                    pass
            total += 1

            if real_list[i].startswith('NIL'):
                nil_total += 1
            if real_list[i]  == 'NIL_Person':
                nil_person_total += 1
            

            if predict_list[i] == real_list[i]:
                correct += 1
            else:
                wrong_list.append(predict_list_all[i])
                wrong_list[-1]['ground_truth'] = real_list[i]
                if real_list[i].startswith('NIL'):
                    nil_wrong += 1
                if real_list[i] == 'NIL_Person':
                    nil_person_wrong += 1

            if real_list[i].startswith('NIL') and (not predict_list[i].startswith('NIL')):
                Wrongcall_num += 1
            if (not real_list[i].startswith('NIL')) and predict_list[i].startswith('NIL'):
                Nocall_num += 1

        print(nil_total)
        print(nil_wrong)
        print(nil_person_total)
        print(nil_person_wrong)

        # pd.DataFrame(wrong_list).to_csv("./output/wrong_list.csv", encoding="gbk")
        # with open("./output/wrong_list.out", "w", encoding='utf-8') as f:
        #     for x in wrong_list:
        #         x["score"] = x["score"].tolist()
        #         f.write(json.dumps(x) + "\n")
        
        return correct, total, NIL_correct, NIL_total, Nocall_num, Wrongcall_num

    def output_result(self, predict_list):
        p = 0
        result = copy.deepcopy(self.datadf)
        for i in range(result.shape[0]):
            mention_data = result['mention_data'][i]
            for mention in mention_data:
                mention['kb_id'] = predict_list[p]
                p += 1
            result['mention_data'][i] = mention_data
        # print(result.head(1))
        result['text_id'] = result['text_id'].apply(str)
        result.to_json('result.json',lines=True,orient='records', date_format='iso')
        return

    def evaluate(self):
        self.readdata()
        if 1:
            if self.eval:
                dataset_1, dataset_2, sentence_list, real_entity_id_list = self.create_test_data(self.KB, self.datadf, self.name_dict)
                save_obj(sentence_list, "./output/evaluate/sentence_list.pkl")
                save_obj(real_entity_id_list, "./output/evaluate/real_entity_id_list.pkl")
            else:
                dataset_1, dataset_2, sentence_list, real_entity_id_list = self.create_test_data(self.KB, self.datadf, self.name_dict)
                save_obj(sentence_list, "./output/test/sentence_list.pkl")
        else:
            if self.eval:
                sentence_list = load_obj("./output/evaluate/sentence_list.pkl")
                real_entity_id_list = load_obj("./output/evaluate/real_entity_id_list.pkl")
            else:
                sentence_list = load_obj("./output/test/sentence_list.pkl")

        if self.re_bert:
            self.base_model.load_state_dict(torch.load("./models/pytorch/base_model.pth"))
            self.type_model.load_state_dict(torch.load("./models/pytorch/type_model.pth"))
            if self.use_gpu:
                self.base_model = self.base_model.cuda()
                self.type_model = self.type_model.cuda()
            
            if self.eval:
                model1_out = self.model_forward(self.base_model, dataset_1, bert_batchsize=768, linear_batchsize=4096, name="1")
                model2_out = self.model_forward(self.type_model, dataset_2, bert_batchsize=512, linear_batchsize=4096,name="2")
            else:
                model1_out = self.model_forward(self.base_model, dataset_1, bert_batchsize=768, linear_batchsize=4096, name="1t")
                model2_out = self.model_forward(self.type_model, dataset_2, bert_batchsize=512, linear_batchsize=4096,name="2t")

            if self.eval:
                save_obj(model1_out, "./output/evaluate/tmp_out1.pkl")
                save_obj(model2_out, './output/evaluate/tmp_out2.pkl')
            else:
                save_obj(model1_out, "./output/test/tmp_out1.pkl")
                save_obj(model2_out, './output/test/tmp_out2.pkl')

            # if self.eval:
            #     model1_out = load_obj('./output/evaluate/tmp_out1.pkl')
            # else:
            #     model1_out = load_obj('./output/test/tmp_out1.pkl')
        else:
            if self.eval:
                model1_out = load_obj('./output/evaluate/tmp_out1.pkl')
                model2_out = load_obj('./output/evaluate/tmp_out2.pkl')
            else:
                model1_out = load_obj('./output/test/tmp_out1.pkl')
                model2_out = load_obj('./output/test/tmp_out2.pkl')

        print(model1_out.shape)
        print(model2_out.shape)

        self.NILtype_model.load_state_dict(torch.load("./models/pytorch/NILtype_model.pth"))
        predict_list = self.get_predict(sentence_list, model1_out, model2_out,self.NILtype_model)


        if self.eval:
            correct, total, NIL_correct, NIL_total, Nocall_num, Wrongcall_num = self.check_predict(predict_list,
                                                                                                    real_entity_id_list)
            print('正确率：')
            print(correct, total)
            print('NIL正确率：')
            print(NIL_correct, NIL_total)
            print('未召回：')  # 尽量百分之百
            print(Nocall_num)
            print('误录：')
            print(Wrongcall_num)
        else:
            self.output_result(predict_list)

    def analyze(self):
        result = []
        low_score = []
        with open('./output/wrong_list.out', "r", encoding='utf-8') as f:
            rd = f.readline()
            while rd:
                x = json.loads(rd)
                x["score"] = x["score"]
                if not x["answer"].startswith("NIL") and x["ground_truth"].startswith("NIL"):
                    if max(x["nil_score"]) > 0.5:
                        result.append(x)
                    else:
                        low_score.append(x)
                rd = f.readline()

        pd.DataFrame(low_score).to_csv("./output/analyze_low.csv", encoding="gbk")

        # result[126: ] = result[127: ]
        # result[375: ] = result[376: ]
        # result[394: ] = result[395: ]
        # pd.DataFrame(result).to_csv("./output/analyze.csv", encoding="gbk")


if __name__ == "__main__":
    # Evaluate(re_bert=False, eval=False).evaluate()
    # Evaluate(re_bert=False, eval=False, type_sel=0).evaluate()
    # Evaluate(re_bert=False, eval=True).analyze()

    parser = argparse.ArgumentParser()
    parser.add_argument("eval", type=bool)
    args = parser.parse_args()

    if args.eval:
        print("evaluate:")
        Evaluate(re_bert=True, eval=True).evaluate()
    else:
        print("test:")
        Evaluate(re_bert=True, eval=False).evaluate()
