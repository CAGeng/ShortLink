import pandas as pd
from tqdm import tqdm
import torch

from lib.models.type_model.model import LinearModel

class TrainFrame:
    def __init__(self, use_gpu, in_dim, n_hidden_1, n_hidden_2, out_dim):
        self.model = LinearModel(in_dim, n_hidden_1, n_hidden_2, out_dim)
        self.use_gpu = use_gpu
        if use_gpu:
            self.model = self.model.cuda()

        self.KB = pd.DataFrame()
        self.traindf = pd.DataFrame()
        self.TYPE_DIC = {}
        self.TYPE_LIST = ['Person', 'Work', 'Other', 'Location', 'Culture', 'VirtualThings',
                          'Organization', 'Game', 'Biological', 'Software', 'Website', 'Food',
                          'Brand', 'Medicine', 'Event', 'Natural&Geography', 'Time&Calendar', 'Vehicle',
                          'Disease&Symptom', 'Education', 'Awards', 'Constellation', 'Diagnosis&Treatment', 'Law&Regulation']

    def readdata(self):
        print("-----------------------读入知识库、训练集、NameDic--------------------------------------")
        self.KB = pd.read_json("./data/baidu/kb.json", lines=True)
        # KB = KB.set_index('subject_id')
        self.traindf = pd.read_json("./data/baidu/train.json", lines=True)
        # self.traindf = self.traindf.head(100)
        # Name_Dic = get_name_dic(KB)
        # Name_Dic = load_obj("Name_Dic")
        for i in range(len(self.TYPE_LIST)):
            self.TYPE_DIC[self.TYPE_LIST[i]] = i

    def create_trainset(self):
        # KB_id = self.KB.set_index('subject_id')
        # traindf = pd.read_json("../input/train.json",lines=True)
        train_data = []
        train_label = []

        label_num = [0 for i in range(30)]

        for i in tqdm(range(self.traindf.shape[0])):
            mention_text = self.traindf['text'][i]
            mention_data = self.traindf['mention_data'][i]
            for data in mention_data:
                entity_id = data['kb_id']
                if not entity_id.startswith("NIL"):
                    continue
                    # entity_type = entity_id[4:]
                entity_type = entity_id[4:]

                if entity_type in self.TYPE_LIST:
                    # continue
                    label = self.TYPE_DIC[entity_type]
                else:
                    # print(entity_type)
                    continue
                    # label = 2  # 'Other'
                train_label.append(label)
                label_num[label] += 1

                begpos = int(data['offset'])
                endpos = begpos + len(data['mention']) - 1

                train_data.append({'text': "[CLS]" + mention_text,
                                   'beg': begpos,
                                   'en': endpos})
        # train_label = torch.tensor(train_label)

        print(label_num)
        # while True:
        #     pass

        return train_data, train_label

    def reget_trainset(self):
        self.readdata()
        print("-----------------------重新获得训练集--------------------------------------")
        train_data, train_label = self.create_trainset()
        train_label = torch.tensor(train_label)
        return train_data, train_label