# 短文本实体链接

## Install

建议使用 conda 管理 python 环境

```powershell
conda create -n shortlinkpj python==3.8
conda activate shortlinkpj
# 创建 conda 环境
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
pip install transformers
pip install matplotlib, pandas, scipy, tqdm
```

## 项目结构

```
${POSE_ROOT}
├── checkpoint
├── data
├── lib
├── models
├── figures
├── output
├── tmp
├── tools 
├── README.md
└── requirements.txt
```

将数据集放在 `data` 文件夹中，命名为 `baidu`，最终数据集目录结构

```
|-- data
`-- |-- baidu
    `-- |-- dev.json
        |-- kb.json
        |-- test.json
        |-- train.json
```

将预训练的 bert  和 tokenizer 参数放入 `lib/models/bert_base_chinese`  文件夹中，最终 ` lib `目录结构如下

```
|-- lib
`-- |-- models
    `-- |-- bert_model
        |-- base_model
        |-- type_model
        |-- NILtype_model
        |-- bert_base_chinese
    |-- dataset
    |-- vote
    |-- __init__.py
    |-- utils.py
```

## 训练

`python tools/train.py`

在 `checkpoint` 目录中输出三个网络的参数 `base_model.pth`，`type_model`，`Niltype_model`

## 验证

将三个网络的参数移动到在 `models/pytorch`  目录下

`python tools/vote.py eval=True`

## 测试

将三个网络的参数移动到在 `models/pytorch`  目录下

`python tools/vote.py eval=False`

最终结果 `result.json` 在项目根目录下