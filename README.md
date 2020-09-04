# mindspore_bert
使用mindspore的bert做finetune

## 1.下载数据
来自https://github.com/CLUEbenchmark/CLUE里面的今日头条新闻数据集
华为对象存储下载预训练模型：/chz/bert_base.ckpt，并上传到自己的对象存储

## 2.预处理：转TFRecord（本地）
第一步：json转tsv
```
python preprocess.py 'train'
python preprocess.py 'dev'

```   
第二步：tsv转TFRecord
```
sh run.sh
```
生成的tfrecord上传到对象存储
    
## 3.finetune（modelarts）
将bert目录上传到对象存储，modelarts执行的代码目录选它
执行finetune.py

## 4.评估（modelarts）
执行evaluation.py
