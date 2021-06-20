import math

import torch.nn as nn

class MLPMoedl(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        x = self.layers(x)
        return x

class ProbModel:
    def __init__(self):
        pass

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
