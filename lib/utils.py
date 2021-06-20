import pandas as pd
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt

def save_obj(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def plot_curve(data):
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['loss_value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()

def entity_recognition(mention, KB, name_dict):
    if mention not in name_dict.keys():
        return None
    candidate_entity_id = name_dict[mention]
    # df = KB.set_index('subject_id')
    candidate_entity = KB.loc[candidate_entity_id, :]
    return candidate_entity

def generate_entity_text(predicate_list, entity_type):
    text = "类型：" + entity_type + "|"
    type_important = {
        'Person':['义项描述','摘要','国籍','职业','出生日期'],
        'Work':['义项描述','摘要','类型','导演','首播时间'],
        'Location':['义项描述','摘要','别名','著名景点','标签'],
        'Culture':['义项描述','摘要','中文名','应用学科','标签'],
        'VirtualThings':['义项描述','摘要','登场作品','中文名','外号'],
        'Organization':['义项描述','摘要','属性','领导人','类别']
    }
    data_dict = {}
    for data in predicate_list:
        predicate = data["predicate"]
        obj = data['object']
        data_dict[predicate] = obj

    if entity_type in type_important.keys():
        important_list = type_important[entity_type]
        for predicate in important_list:
            if predicate in data_dict.keys():
                text += data_dict[predicate]
        for predicate in data_dict.keys():
            if predicate not in important_list:
                text += data_dict[predicate]
    else:
        important_list = ['义项描述','摘要']
        for predicate in important_list:
            if predicate in data_dict.keys():
                text += data_dict[predicate]
        for predicate in data_dict.keys():
            if predicate not in important_list:
                text += data_dict[predicate]
    return text

def generate_text(mention_text, entity_text):
    text = "[CLS] " + mention_text + " [SEP] " + entity_text + " [SEP] "
    return text

def get_name_dic(KB):
    Name_Dic = {}
    print('----------------------get NameDic--------------------------------------')
    for i in tqdm(range(KB.shape[0])):
        name = KB['subject'][i]
        if name not in Name_Dic.keys():
            Name_Dic[name] = []
        Name_Dic[name].append(KB['subject_id'][i])

        aliaslist = KB['alias'][i]
        for name in aliaslist:
            if name not in Name_Dic.keys():
                Name_Dic[name] = []
            Name_Dic[name].append(KB['subject_id'][i])
    return Name_Dic