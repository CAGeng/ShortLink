from tqdm import tqdm
import math
import pandas as pd
import numpy as np

class Vote:
    def __init__(self, POPULARITY_DIC = {}, KB = {}, traindf = pd.DataFrame()):
        self.KB = KB
        self.POPULARITY_DIC = POPULARITY_DIC
        self.traindf = traindf

    def readdata(self):
        print("-----------------------读入--------------------------------------")
        self.KB = pd.read_json("../input/kb.json", lines=True)
        self.traindf = pd.read_json("../input/train.json", lines=True)

    def get_Popularity(self):
        print('----------------------get POPULARITY_DIC--------------------------------------')
        mention_sum = {}
        for i in tqdm(range(traindf.shape[0])):
            text = traindf['text'][i]
            mention_data = traindf['mention_data'][i]
            for data in mention_data:
                entity_id = data['kb_id']
                begpos = int(data['offset'])
                endpos = begpos + len(data['mention'])
                mention_text = text[begpos:endpos]

                if mention_text not in self.POPULARITY_DIC.keys():
                    self.POPULARITY_DIC[mention_text] = {}
                if entity_id not in self.POPULARITY_DIC[mention_text].keys():
                    self.POPULARITY_DIC[mention_text][entity_id] = 0
                self.POPULARITY_DIC[mention_text][entity_id] += 1
                if mention_text not in mention_sum.keys():
                    mention_sum[mention_text] = 0
                mention_sum[mention_text] += 1

        for mention in self.POPULARITY_DIC.keys():
            sum = mention_sum[mention]
            for entity in self.POPULARITY_DIC[mention].keys():
                self.POPULARITY_DIC[mention][entity] /= sum

    def add_common_appear(self, entity1, entity2):
        global COH_DIC
        if entity1 not in COH_DIC.keys():
            COH_DIC[entity1] = {}
        if entity2 not in COH_DIC[entity1].keys():
            COH_DIC[entity1][entity2] = 0
        COH_DIC[entity1][entity2] += 1

    def add_ENTITY_CON(self, entity):
        global ENTITY_CON, ENTITY_CON_TOTAL
        if entity not in ENTITY_CON.keys():
            ENTITY_CON[entity] = 0
        ENTITY_CON[entity] += 1
        ENTITY_CON_TOTAL += 1

    def get_coherence(self):
        global traindf
        print('----------------------get COH--------------------------------------')
        for i in tqdm(range(traindf.shape[0])):
            mention_data = traindf['mention_data'][i]
            entid_list = []
            for data in mention_data:
                entity_id = data['kb_id']
                self.add_ENTITY_CON(entity_id)
                entid_list.append(entity_id)
            # for en1 in entid_list:
            #     for en2 in entid_list:
            #         add_common_appear(en1,en2)
            for i in range(len(entid_list)):
                for j in range(i + 1, len(entid_list)):
                    en1 = entid_list[i]
                    en2 = entid_list[j]
                    self.add_common_appear(en1, en2)
                    self.add_common_appear(en2, en1)

    def calculate_NGD(self, E1, E2, E, E1E2):
        if E1 == 0 or E2 == 0 or E1E2 == 0:
            return 0
        t1 = math.log(max(E1, E2))
        t2 = math.log(E1E2)
        t3 = math.log(E)
        t4 = math.log(min(E1, E2))
        # print(1 - (t1 - t2) / (t3 - t4))
        return 1 - (t1 - t2) / (t3 - t4)
        # if E1 + E2 == 0:
        #     return 0
        # return (E1E2) / (E1 + E2)

    def get_coh_e1e2(self, e1, e2, ENTITY_CON, ENTITY_CON_TOTAL, COH_DIC):
        # global ENTITY_CON,ENTITY_CON_TOTAL,COH_DIC
        # print('con' + str(ENTITY_CON_TOTAL))

        # 相同实体
        if e1 == e2:
            return 0.9
        if e1 not in ENTITY_CON.keys():
            E1 = 0
        else:
            E1 = ENTITY_CON[e1]
        if e2 not in ENTITY_CON.keys():
            E2 = 0
        else:
            E2 = ENTITY_CON[e2]
        if e1 not in COH_DIC.keys() or e2 not in COH_DIC[e1].keys():
            E1E2 = 0
        else:
            E1E2 = COH_DIC[e1][e2]
        return self.calculate_NGD(E1, E2, ENTITY_CON_TOTAL, E1E2)