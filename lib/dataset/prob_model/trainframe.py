import pandas as pd
from tqdm import tqdm
import torch

from lib.utils import *
from lib.vote.vote import Vote as vote
from lib.models.type_model.model import LinearModel as type_model

KEYS_H7 = ['Person', 'Work', 'Location', 'Culture', 'VirtualThings', 'Organization', 'Game']

class TrainFrame:
    def __init__(self):
        self.KB = pd.DataFrame()
        self.datadf = pd.DataFrame()
        self.name_dict = {}
        self.ENTITY_CON = {}
        self.ENTITY_CON_TOTAL = {}
        self.COH_DIC = {}
        self.POPULARITY_DIC = {}

        self.vote = vote(self.POPULARITY_DIC, self.KB, self.datadf)
        pass

    def readdata(self):
        print("-----------------------读入--------------------------------------")
        # 基础
        self.KB = pd.read_json("./input/kb.json", lines=True)
        self.datadf = pd.read_json("./input/dev.json", lines=True)
        # self.datadf = datadf.head(100)
        self.Name_Dic = load_obj("Name_Dic")

        # 统计数据
        self.ENTITY_CON = load_obj("ENTITY_CON")
        self.ENTITY_CON_TOTAL = load_obj("ENTITY_CON_TOTAL")
        self.COH_DIC = load_obj("COH_DIC")
        self.POPULARITY_DIC = load_obj("POPULARITY_DIC")

        # 类型（上位概念）
        for i in range(len(KEYS_H7)):
            self.KEY_DIC[KEYS_H7[i]] = i
        self.vote = vote(self.POPULARITY_DIC, self.KB, self.datadf)

    def reget_trainset(self):
        self.readdata()
        self.
